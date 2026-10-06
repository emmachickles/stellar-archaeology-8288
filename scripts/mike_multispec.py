"""Read IRAF multispec MIKE files, continuum-normalise orders, merge to 1D.

The course gives raw reduced `*_multi.fits` (bands x orders x pixels) with an IRAF
`WAT2_*` multispec wavelength solution.  Worksheet 1 assumed an already merged,
normalised spectrum, so this rebuilds that product for our own star.
"""
import numpy as np
from astropy.io import fits


def _wat(header, key="WAT2"):
    """Concatenate WAT cards.  IRAF pads each card to 68 chars; astropy strips
    trailing blanks, so re-pad before joining or tokens get glued together."""
    out = []
    i = 1
    while f"{key}_{i:03d}" in header:
        out.append(header[f"{key}_{i:03d}"].ljust(68))
        i += 1
    return "".join(out)


def read_multispec(path, band="object spectrum divided by normed flat"):
    """-> list of (wave, flux, snr) per order, plus the header."""
    hdu = fits.open(path)[0]
    hdr, cube = hdu.header, hdu.data
    bands = {hdr[k].strip(): int(k[6:]) - 1 for k in hdr if k.startswith("BANDID")}
    bi = bands[band]
    si = bands.get("signal-to-noise spectrum")

    txt = _wat(hdr)
    orders = []
    for chunk in txt.split('spec')[1:]:
        q = chunk.split('"')
        if len(q) < 2:
            continue
        p = q[1].split()
        ap, beam, dtype = int(p[0]), int(p[1]), int(p[2])
        w1, dw, nw = float(p[3]), float(p[4]), int(p[5])
        if dtype != 0:
            raise ValueError(f"{path}: non-linear dispersion (dtype={dtype}) not handled")
        idx = len(orders)
        w = w1 + dw * np.arange(nw)
        f = np.asarray(cube[bi, idx, :nw], float)
        s = np.asarray(cube[si, idx, :nw], float) if si is not None else np.full(nw, np.nan)
        orders.append((w, f, s))
    return orders, hdr


def despike(f, width=7, nsig=6.0):
    """Clip single-pixel cosmic rays against a running median.

    Only spikes *upward* are removed: a cosmic ray adds flux, whereas a real
    absorption line removes it, so clipping downward would eat line cores.
    """
    f = np.asarray(f, float)
    n = len(f)
    if n < width:
        return f
    pad = width // 2
    padded = np.pad(f, pad, mode="edge")
    med = np.array([np.median(padded[i:i + width]) for i in range(n)])
    resid = f - med
    # robust scatter: a plain std is inflated by the deep Balmer cores, which
    # then hides single hot pixels behind a too-large sigma
    s = 1.4826*np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    if not np.isfinite(s) or s == 0:
        return f
    out = f.copy()
    bad = resid > nsig * s
    out[bad] = med[bad]
    return out


def normalise_order(w, f, niter=6, lo=1.5, hi=3.0, deg=4, minpts=40, mask=None):
    """Asymmetric sigma-clipped polynomial fit to the upper envelope.

    Absorption lines only ever push flux *down*, so clip hard below the fit and
    loosely above it; that pulls the solution onto the continuum instead of the
    mean.  Returns normalised flux and the continuum.
    """
    g = np.isfinite(f) & (f > 0)
    if mask is not None:
        # Broad lines (the Balmer series) are wider than the continuum windows
        # left in a short echelle order, so a free fit curves down into their
        # wings.  Exclude those regions from the fit but keep them in the output.
        for a, b in mask:
            g &= ~((w > a) & (w < b))
    if g.sum() < minpts:
        return np.full_like(f, np.nan), np.full_like(f, np.nan)
    x = np.linspace(-1, 1, len(w))
    m = g.copy()
    c = np.full_like(f, np.nan)
    for _ in range(niter):
        if m.sum() < minpts:
            break
        p = np.polyfit(x[m], f[m], deg)
        c = np.polyval(p, x)
        r = f - c
        s = np.std(r[m])
        if not np.isfinite(s) or s == 0:
            break
        m = g & (r > -lo * s) & (r < hi * s)
    with np.errstate(invalid="ignore", divide="ignore"):
        return f / c, c


BALMER_MASK = [(4090, 4115), (4325, 4356), (4845, 4880), (6545, 6580), (3960, 3985)]


def merge_orders(orders, dw=None, snr_floor=1.0, min_snr=8.0, edge_trim=0.05,
                 mask=None, clip_spikes=True, deg=4):
    """S/N^2-weighted merge onto a common linear grid.

    Orders whose median S/N is below `min_snr` are dropped outright -- at the
    MIKE blue cutoff they are pure noise and the continuum fit is meaningless.
    `edge_trim` removes the order ends, where the blaze fit is least reliable.
    """
    orders = [o for o in orders if np.nanmedian(o[2]) >= min_snr]
    if not orders:
        raise ValueError("no orders above min_snr")
    ws = [o[0] for o in orders]
    w_lo = min(w.min() for w in ws)
    w_hi = max(w.max() for w in ws)
    if dw is None:
        dw = np.median([np.median(np.diff(w)) for w in ws])
    grid = np.arange(w_lo, w_hi + dw, dw)
    num = np.zeros_like(grid)
    den = np.zeros_like(grid)
    for w, f, s in orders:
        if clip_spikes:
            f = despike(f)
        # only mask regions that leave enough continuum in this order to fit
        om = None
        if mask:
            inside = [(a, b) for a, b in mask if b > w[0] and a < w[-1]]
            covered = sum(min(b, w[-1]) - max(a, w[0]) for a, b in inside)
            if inside and covered < 0.55 * (w[-1] - w[0]):
                om = inside
        nf, _ = normalise_order(w, f, mask=om, deg=deg)
        good = np.isfinite(nf) & np.isfinite(w)
        if good.sum() < 40:
            continue
        # trim the order ends, where the blaze fit is worst
        k = int(edge_trim * good.sum())
        idx = np.where(good)[0][k:len(np.where(good)[0]) - k]
        if len(idx) < 40:
            continue
        wt = np.clip(np.nan_to_num(s[idx], nan=snr_floor), snr_floor, None) ** 2
        fi = np.interp(grid, w[idx], nf[idx], left=np.nan, right=np.nan)
        wi = np.interp(grid, w[idx], wt, left=0.0, right=0.0)
        ok = np.isfinite(fi)
        num[ok] += fi[ok] * wi[ok]
        den[ok] += wi[ok]
    with np.errstate(invalid="ignore", divide="ignore"):
        flux = num / den
    keep = den > 0
    return grid[keep], flux[keep]


def der_snr(flux):
    """DER_SNR estimator (Stoehr et al. 2008).

    Uses the pixel-to-pixel difference on a 5-pixel stencil, so it is insensitive
    to the slow curvature of line wings and needs no line-free window.
    """
    f = np.asarray(flux, float)
    f = f[np.isfinite(f)]
    if len(f) < 10:
        return np.nan
    signal = np.median(f)
    noise = 1.482602/np.sqrt(6.0) * np.median(
        np.abs(2.0*f[2:-2] - f[:-4] - f[4:]))
    return np.nan if noise == 0 else signal/noise


C_KMS = 299792.458


def coadd_rest(epochs, dw=0.05, w_lo=None, w_hi=None):
    """Shift each epoch to rest by its own velocity and stack, S/N^2 weighted.

    `epochs` is a list of (wave, flux, v_geo_kms, weight_snr).  Coadding is what
    makes the faint blue lines usable: one 1800 s exposure does not reach the
    S/N that Eu, Th and Pb need.
    """
    rest = [(w/(1.0 + v/C_KMS), f, wt) for w, f, v, wt in epochs]
    # union, not intersection: taking the intersection lets the epoch with the
    # narrowest coverage truncate the blue end and throw away Th 4019 / Pb 4057.
    # Pixels covered by fewer epochs simply carry less weight.
    lo = w_lo if w_lo is not None else min(r[0].min() for r in rest)
    hi = w_hi if w_hi is not None else max(r[0].max() for r in rest)
    grid = np.arange(lo, hi + dw, dw)
    num = np.zeros_like(grid)
    den = np.zeros_like(grid)
    for w, f, wt in rest:
        fi = np.interp(grid, w, f, left=np.nan, right=np.nan)
        ok = np.isfinite(fi)
        num[ok] += fi[ok]*wt**2
        den[ok] += wt**2
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num/den
    keep = den > 0
    return grid[keep], out[keep]
