"""Ca II H+K and Mg I 4167 for every individual exposure, in the OBSERVED frame.

No heliocentric correction and no shift to rest: the wavelength scale is exactly
as the pipeline delivered it. HD 122563, which is already at rest, is overplotted
so the displacement is visible -- it is the star's full geocentric velocity, and
it changes from exposure to exposure as the Earth moves.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from mike_multispec import read_multispec, normalise_order, despike, der_snr, C_KMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
D = "data/he1002-0755_epochs"
OUT = "worksheets/ws2/part3_summary_plot/plots"

EXPOSURES = [  # (label, blue, red) -- all eleven, in time order
 ("2009-02-05 05:43", "he1002-0755ablue_multi_ut090205_07slit.fits", "he1002-0755ared_multi_ut090205_07slit.fits"),
 ("2009-02-06 02:38", "he1002-0755bblue_multi_ut090205_07slit.fits", "he1002-0755bred_multi_ut090205_07slit.fits"),
 ("2009-02-07 03:58", "he1002-0755cblue_multi_ut090205_07slit.fits", "he1002-0755cred_multi_ut090205_07slit.fits"),
 ("2010-05-08 22:44", "he1002-0755_2010_blue_multi.fits", "he1002-0755_2010_red_multi.fits"),
 ("2011-03-10 00:31", "he1002-0755blue_multi_ut110310_07slit.fits", "he1002-0755red_multi_ut110310_07slit.fits"),
 ("2013-05-30 00:54", "he1002-0755blue_multi_ut130530_07slit.fits", "he1002-0755red_multi_ut130530_07slit.fits"),
 ("2013-05-30 22:50", "he1002-0755_2013_may30_blue_multi.fits", "he1002-0755_2013_may30_red_multi.fits"),
 ("2013-05-31 23:09", "he1002-0755_2013_may31_blue_multi.fits", "he1002-0755_2013_may31_red_multi.fits"),
 ("2014-03-12 00:34", "he1002-0755_2014_blue_multi.fits", "he1002-0755_2014_red_multi.fits"),
 ("2017-06-05 00:11", "he1002-0755_2017_blue_multi.fits", "he1002-0755_2017_red_multi.fits"),
 ("2018-03-07 03:52", "he1002-0755_2018_blue_multi.fits", "he1002-0755_2018_red_multi.fits"),
]
# Velocities come from results/rv_lines_per_exposure.json (mean over the lines
# measurable in that exposure, with the line-to-line standard error). Two
# quantities, two jobs:
#   v_geo   -- where the line actually sits in this uncorrected spectrum, so it
#              is what the dashed marker must use;
#   v_helio -- the star's velocity with the Earth's motion removed, which is the
#              quantity an RV curve is made of, so it is what gets annotated.
_RV = json.load(open("worksheets/ws2/part2_observation_details/results/rv_lines_per_exposure.json"))
VGEO, VHELIO, VERR = {}, {}, {}
for _r in _RV:
    _t = _r["exposure"]
    VHELIO[_t] = _r["v_helio"]
    VERR[_t] = _r["v_helio_err"]
    VGEO[_t] = None if _r["v_helio"] is None else _r["v_helio"] - _r["helcorr"]

REGIONS = {
  "CaHK":   dict(lo=3925.0, hi=3980.0, marks={"Ca II K": 3933.66, "Ca II H": 3968.47},
                 label="Ca II H and K", ylim=(-0.05, 1.35)),
  "Mg4167": dict(lo=4157.0, hi=4178.0, marks={"Mg I 4167": 4167.27},
                 label="Mg I 4167", ylim=(0.35, 1.25)),
}

def path(n):
    p = os.path.join(D, n)
    return p if os.path.exists(p) else os.path.join("data", n)

def ref_spectrum():
    h = fits.open("data/HD122563.fits")[0]; hd = h.header
    n = hd["NAXIS1"]; cd = hd.get("CD1_1", hd.get("CDELT1"))
    return hd["CRVAL1"] + cd*(np.arange(n) + 1 - hd.get("CRPIX1", 1)), np.asarray(h.data, float)

def order_for(orders, lo, hi):
    """Highest-S/N order fully covering the window."""
    c = [o for o in orders if o[0][0] < lo - 3 and o[0][-1] > hi + 3]
    return max(c, key=lambda o: np.nanmedian(o[2])) if c else None

def main(raw=False):
    """raw=True: plot the extracted order exactly as the pipeline delivered it --
    no despiking, no continuum normalisation, no S/N or velocity annotation."""
    wr, fr = ref_spectrum()
    plt.rcParams.update({"font.family": "serif", "font.size": 8,
        "axes.labelsize": 10, "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "legend.frameon": False})

    # BANDID7 ("object / normed flat") for the processed sheets; BANDID2
    # ("object spectrum") for the raw ones. BANDID7 is already flat-divided, so
    # most of the blaze is gone from it -- plotting it would not show the raw
    # instrument response. SMHR also loads BANDID2.
    band = "object spectrum" if raw else "object spectrum divided by normed flat"
    cache = {}
    for tag, b, r in EXPOSURES:
        ords = []
        for f in (path(b), path(r)):
            o, _ = read_multispec(f, band=band); ords += o
        cache[tag] = ords

    for rname, R in REGIONS.items():
        n = len(EXPOSURES)
        fig, axes = plt.subplots(n, 1, figsize=(9.0, 1.05*n + 1.2), sharex=True,
                                 gridspec_kw=dict(hspace=0.0))
        for ax, (tag, b, r) in zip(axes, EXPOSURES):
            v = VGEO[tag]
            m = (wr > R["lo"] - 2) & (wr < R["hi"] + 2)
            if not raw:
                # HD 122563, at rest, as the fixed reference
                ax.plot(wr[m], fr[m], color="#0072B2", lw=0.7, alpha=0.8)
            o = order_for(cache[tag], R["lo"], R["hi"])
            snr = np.nan
            if o is not None:
                w, fl, s = o
                g = (w > R["lo"] - 2) & (w < R["hi"] + 2)
                if raw:
                    # exactly as extracted: same band and same order, but no
                    # despiking and no continuum fit. Counts, so it needs its
                    # own axis -- HD 122563 is normalised and cannot share one.
                    axr = ax.twinx()
                    axr.plot(w[g], fl[g], "k-", lw=0.9)
                    # the reference is normalised and our trace is in counts, so
                    # they cannot share a scale: multiply HD 122563 by the median
                    # counts in this window. Our data is untouched.
                    scale = np.nanmedian(fl[g]) if g.sum() else 1.0
                    axr.plot(wr[m], fr[m]*scale, color="#0072B2", lw=0.7, alpha=0.8)
                    # view limits from robust percentiles: nothing is altered,
                    # but a single un-clipped cosmic ray cannot flatten a panel
                    if g.sum():
                        lo_y, hi_y = np.nanpercentile(fl[g], [0.5, 99.0])
                        pad = 0.25*(hi_y - lo_y) if hi_y > lo_y else 1.0
                        axr.set_ylim(min(lo_y - pad, 0.0), hi_y + pad)
                    axr.set_yticks([])
                    axr.set_zorder(ax.get_zorder() + 1); ax.patch.set_visible(False)
                else:
                    # mask only around each line (observed frame), so the continuum
                    # still has most of the order to fit; masking the whole plotted
                    # window leaves a cubic with ~26 A of leverage and it sags
                    zz = 1 + (v if v is not None else 225.0)/C_KMS
                    mask = [(lam*zz - 7, lam*zz + 7) for lam in R["marks"].values()]
                    nf, _ = normalise_order(w, despike(fl), mask=mask, deg=3)
                    g = g & np.isfinite(nf)
                    ax.plot(w[g], nf[g], "k-", lw=1.0)
                    snr = der_snr(nf[g])
            # rest positions (solid) and where they land at this exposure's v_geo (dashed)
            for nm, lam in R["marks"].items():
                ax.axvline(lam, color="#0072B2", ls="-", lw=0.6, alpha=0.5)
                if v is not None and not raw:
                    ax.axvline(lam*(1 + v/C_KMS), color="k", ls="--", lw=0.7, alpha=0.6)
            ax.set_xlim(R["lo"], R["hi"])
            ax.set_ylim(0, 1.3) if raw else ax.set_ylim(*R["ylim"])
            ax.set_yticks([])
            bbox = dict(fc="white", ec="none", alpha=1.0 if raw else 0.85, pad=1.4)
            ax.text(0.006, 0.80, tag, transform=ax.transAxes, fontsize=7.5,
                    family="monospace", va="center", bbox=bbox, zorder=20)
            if not raw:
                vh, ve = VHELIO[tag], VERR[tag]
                vtxt = ("no RV" if vh is None
                        else r"$v_{\rm helio}=%+.2f \pm %.2f$" % (vh, ve))
                ax.text(0.006, 0.09, vtxt, transform=ax.transAxes, fontsize=7,
                        va="center", bbox=bbox)
                ax.text(0.994, 0.09, ("S/N %.0f" % snr) if snr == snr else "S/N --",
                        transform=ax.transAxes, fontsize=7, ha="right", va="center", bbox=bbox)
        axes[-1].set_xlabel(r"Observed wavelength, $\lambda$ ($\AA$)")
        for nm, lam in R["marks"].items():
            axes[0].annotate(nm, xy=(lam, 1.0), xycoords=("data", "axes fraction"),
                             xytext=(0, 4), textcoords="offset points",
                             ha="center", fontsize=8, color="#0072B2")
        if raw:
            fig.suptitle(
                "%s in each of the %d MIKE exposures of HE 1002-0755\n"
                "Observed frame, uncorrected. HD 122563 (blue, scaled to the "
                "median counts) is at rest. Band: object spectrum, not flat-divided."
                % (R["label"], n),
                y=0.998, fontsize=10, linespacing=1.7)
        else:
            fig.suptitle("%s in each of the %d MIKE exposures of HE 1002-0755"
                         % (R["label"], n), y=0.997, fontsize=11)
            fig.text(0.5, 0.968, "Observed frame, uncorrected. HD 122563 (blue) is at rest; solid "
                     "lines mark rest wavelengths, dashed lines the observed positions.\n"
                     "Annotated velocities are heliocentric: the lines move between exposures, "
                     "the star's velocity does not.", ha="center", fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.955])
        stem = "per_exposure_%s%s" % (rname, "_raw" if raw else "")
        for ext in ("png", "pdf"):
            fig.savefig("%s/%s.%s" % (OUT, stem, ext), dpi=170, bbox_inches="tight")
        plt.close(fig)
        print("wrote %s/%s.{png,pdf}" % (OUT, stem))

if __name__ == "__main__":
    main(raw=False)
    main(raw=True)
