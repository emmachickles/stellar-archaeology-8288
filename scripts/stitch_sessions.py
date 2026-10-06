#!/usr/bin/env python
"""Normalise, RV-correct and stitch every HE 1002-0755 SMHR session, using SMHR's
own code for each step, and write the products to data/stitched/.

Run in the smhr-py3 env:
    conda activate smhr-py3 && cd ~/software/smhr
    python scripts/stitch_sessions.py --validate
    python scripts/stitch_sessions.py --run

Nothing in data/smh_sessions/ is ever written. In particular the 2009-02-05
session may hold work saved from the GUI; it is only read.

What is SMHR's and what is ours
  * radial velocity : Session.rv_measure (SMHR), template hd122563, window below
  * continuum       : Spectrum1D.fit_continuum (SMHR), the same settings the GUI
                      applies -- see NORM below
  * stitching       : Session.stitch_and_stack (SMHR)
Only the loop that calls them is ours, which stands in for clicking through the
GUI once per exposure.

--validate reproduces the recipe on the 2009-02-05 session and compares it with
the stitched spectrum saved from the GUI, so it is checked against the GUI
rather than trusted.
"""
import os, sys, glob, csv, argparse, logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import yaml
import smh
from smh import specutils
from smh.smh_plotting import make_summary_plot

logging.getLogger().setLevel(logging.ERROR)
for _n in ("smh", "smh.session", "smh.specutils"):
    logging.getLogger(_n).setLevel(logging.ERROR)

ROOT = os.environ.get("COURSE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SESS = f"{ROOT}/data/smh_sessions"
OUT = f"{ROOT}/data/stitched"
TEMPLATE = os.path.join(os.path.dirname(smh.__file__), "data", "spectra", "hd122563.fits")

# SMHR's default cross-correlation windows (default_session.yaml), in the order
# the GUI lists them.
RV_WINDOWS = [(8450., 8750.), (5100., 5200.), (6510., 6610.), (4810., 4910.), (4290., 4390.)]
# The window whose velocity is applied: the Ca triplet region. It is the window
# Worksheet 3 names, it is the one used in the GUI session, and it has by far the
# best order (S/N 90-160 against 22-30 at Mg b). Every SMHR window is recorded in
# rv_smhr_windows.csv, so the choice can be revisited.
#
# CORRECTION (2026-09-29): an earlier analysis of mine called the triplet unusable,
# biased by up to 9-13 km/s at high geocentric velocity. That was measured with
# narrow +-8 A windows around each line. SMHR's 300 A window shows nothing of the
# kind: Ca triplet minus Mg b is -0.2 to -2.2 km/s at every geocentric velocity
# from +216 to +253. The bias is specific to narrow single-line windows.
ADOPTED = (8450., 8750.)
# Hbeta (the order is only 100 A wide) and Halpha (chromospheric core) are
# recorded but never used to judge whether a detection is real.
MIN_SNR = 3.0  # only used to decide which exposures define the reference velocity


# The settings the GUI applied to all 70 orders in the 2009-02-05 session.
NORM = dict(function="Spline", order=3, low_sigma_clip=1.0, high_sigma_clip=0.2,
            knot_spacing=100.0, max_iterations=5)
BLUE_TRIM, RED_TRIM = 50, 20             # pixels
GLOBAL_MASK = [(7593., 7618.)]           # telluric A band, observed frame (GUI "default" mask)


def stamp_of(path):
    return os.path.basename(path)[:-4]


# --------------------------------------------------------------------------- RV
def measure_rvs(s):
    """SMHR cross-correlation in every window, the adopted one LAST so that
    session.metadata['rv'] describes it when rv_correct is called."""
    windows = [w for w in RV_WINDOWS if w != ADOPTED] + [ADOPTED]
    rows = []
    for lo, hi in windows:
        try:
            rv, unc = s.rv_measure(template_spectrum=TEMPLATE, wavelength_region=(lo, hi))
            m = s.metadata["rv"]
            idx = int(m["order_index"])
            o = s.input_spectra[idx]
            with np.errstate(all="ignore"):
                snr = float(np.nanmedian(o.flux * np.sqrt(o.ivar)))
            rows.append(dict(lo=lo, hi=hi, rv=float(rv), ccf_width=float(unc), order=idx,
                             order_snr=snr, hc=float(m["heliocentric_correction"]),
                             bc=float(m["barycentric_correction"]), err=""))
        except Exception as e:
            rows.append(dict(lo=lo, hi=hi, rv=np.nan, ccf_width=np.nan, order=-1,
                             order_snr=np.nan, hc=np.nan, bc=np.nan, err=repr(e)[:80]))
    return rows


def campaign_reference(all_rows):
    """Median barycentric-frame velocity over the exposures whose adopted order has
    real signal. Used only to reject cross-correlations that latched onto noise --
    it never enters the velocity that is applied.  The star's velocity is constant
    to a few km/s across the campaign, so a tolerance of 20 km/s is very loose."""
    v = []
    for rows in all_rows:
        ad = [r for r in rows if (r["lo"], r["hi"]) == ADOPTED][0]
        if np.isfinite(ad["rv"]) and ad["order_snr"] >= MIN_SNR:
            v.append(ad["rv"] + ad["hc"])
    return float(np.median(v))


def rv_verdict(rows, ref, tol=20.0):
    """(valid, note).

    An exposure is valid when the adopted window's velocity, taken to the
    heliocentric frame, lies within `tol` of the reference. This replaces an
    order-S/N floor, which wrongly rejected 2013-05-31: the Mg b order there has
    S/N 1.2, yet the cross-correlation still found the star, and the Ca triplet,
    H-alpha and Mg b all agree at +253.
    """
    ad = [r for r in rows if (r["lo"], r["hi"]) == ADOPTED][0]
    if not np.isfinite(ad["rv"]):
        return False, "adopted window failed: " + ad["err"]
    v = ad["rv"] + ad["hc"]
    others = [r["rv"] for r in rows if (r["lo"], r["hi"]) != ADOPTED and np.isfinite(r["rv"])]
    agree = sum(abs(o - ad["rv"]) < 5.0 for o in others)
    note = "%d of %d other windows within 5 km/s" % (agree, len(others))
    if abs(v - ref) > tol:
        return False, "%+.1f is %.0f km/s from the campaign reference %+.1f: not the star (%s)" % (
            v, abs(v - ref), ref, note)
    return True, note


# ------------------------------------------------------------------- continuum
def normalise_all(s):
    """Fit every order's continuum with SMHR's fit_continuum, reproducing what the
    GUI does.  Returns [(order, reason)] for orders where no continuum could be fit.

    Two details of the GUI that matter, both established by reproducing a
    GUI-made session to floating-point precision:

    1. It fits in a wavelength frame scaled by (1 - rv_applied/c) -- the trim and
       the telluric mask are written in that frame too. Fitting in the observed
       frame with the same pixels masked differs only at the 1e-3 level, but the
       exact frame is free to reproduce, so we do.
    2. The red trim covers the LAST pixel. The interval's end must therefore lie
       beyond x[-1]: searchsorted(x, x[-1]) is n-1 and leaves the final pixel in
       the fit. On orders that end on a steep blaze edge or on noise, that single
       pixel shifts the sigma-clipped spline by 2-3 %.
    """
    c = 299792.458
    f = 1.0 - s.metadata["rv"]["rv_applied"] / c
    n = s.metadata["normalization"]
    failed = []
    for i, order in enumerate(s.input_spectra):
        x = order.dispersion * f
        shifted = specutils.Spectrum1D(x, order.flux, order.ivar)
        excl = [(x[0], x[min(BLUE_TRIM, len(x) - 1)]), (x[-RED_TRIM], x[-1] + 1e-3)]
        for a, b in GLOBAL_MASK:
            a2, b2 = a * f, b * f
            if b2 >= x[0] and a2 <= x[-1]:
                excl.append((a2, b2))
        kw = dict(NORM, exclude=np.array(excl), blue_trim=BLUE_TRIM, red_trim=RED_TRIM,
                  full_output=True)
        try:
            _, cont, _, _ = shifted.fit_continuum(**kw)
        except Exception as e:
            cont = np.full(x.shape, np.nan)
            failed.append((i, repr(e)[:60]))
        n["continuum"][i] = cont
        n["normalization_kwargs"][i] = kw
    return failed


# -------------------------------------------------------------------- validate
def validate():
    path = f"{SESS}/he1002-0755_20090205_0543.smh"
    u = smh.Session.load(path)            # the GUI's result, read only
    m = smh.Session.load(path)            # our recipe applied to the same inputs
    print("GUI session: rv_measured %.4f  rv_applied %.2f  window %s  template %s" % (
        u.metadata["rv"]["rv_measured"], u.metadata["rv"]["rv_applied"],
        u.metadata["rv"]["wavelength_region"],
        os.path.basename(str(u.metadata["rv"].get("template_spectrum_path")))))

    failed = normalise_all(m)             # keeps the GUI's rv_applied, isolating the continuum
    m.stitch_and_stack()

    print("\n--- continuum, order by order (central 80 % of each order) ---")
    worst = []
    for i in range(len(m.input_spectra)):
        cu = np.asarray(u.metadata["normalization"]["continuum"][i], float)
        cm = np.asarray(m.metadata["normalization"]["continuum"][i], float)
        k = np.arange(len(cu))
        mid = (k > 0.1 * len(cu)) & (k < 0.9 * len(cu)) & np.isfinite(cu) & np.isfinite(cm) & (cu > 0)
        if mid.sum() < 20:
            continue
        d = np.abs(cm[mid] / cu[mid] - 1.0)
        worst.append((float(np.max(d)), float(np.median(d)), i,
                      float(m.input_spectra[i].dispersion[0]), float(m.input_spectra[i].dispersion[-1])))
    mx = np.array([w[0] for w in worst]); md = np.array([w[1] for w in worst])
    print("orders compared: %d   median of per-order median |ratio-1| = %.2e" % (len(worst), np.median(md)))
    print("per-order MAX |ratio-1|:  median %.2e   95th pct %.2e   max %.2e" % (
        np.median(mx), np.percentile(mx, 95), mx.max()))
    bad = sorted([w for w in worst if w[0] > 0.02], reverse=True)
    print("orders differing by > 2 %% somewhere in the interior: %d" % len(bad))
    for w in bad[:8]:
        print("   order %2d (%.0f-%.0f A)  max %.1f %%  median %.2f %%" % (w[2], w[3], w[4], 100 * w[0], 100 * w[1]))
    if failed:
        print("orders with no continuum:", failed[:6])

    print("\n--- stitched spectrum vs the GUI's ---")
    fu, wu = u.normalized_spectrum.flux, u.normalized_spectrum.dispersion
    fm = np.interp(wu, m.normalized_spectrum.dispersion, m.normalized_spectrum.flux)
    print("   wavelength grid identical: %s  (len %d vs %d)" % (
        len(wu) == len(m.normalized_spectrum.dispersion) and
        np.allclose(wu, m.normalized_spectrum.dispersion), len(wu), len(m.normalized_spectrum.dispersion)))
    for lo, hi in [(3800, 4500), (4500, 6000), (6000, 8000), (8000, 9100)]:
        g = (wu > lo) & (wu < hi) & np.isfinite(fu) & np.isfinite(fm)
        d = np.abs(fm[g] - fu[g])
        print("   %4d-%4d A   median |diff| %.4f   95th %.4f   99th %.4f   >0.02: %5.1f %%   (n=%d)" % (
            lo, hi, np.median(d), np.percentile(d, 95), np.percentile(d, 99), 100 * np.mean(d > 0.02), g.sum()))

    print("\n--- SMHR's own RV, this exposure, every default window ---")
    r = smh.Session.load(path)
    rows = measure_rvs(r)
    for x in sorted(rows, key=lambda z: z["lo"]):
        print("   %5.0f-%5.0f A   rv %+9.3f   order %2d  order S/N %5.1f   %s" % (
            x["lo"], x["hi"], x["rv"], x["order"], x["order_snr"], x["err"]))
    print("   GUI used 8450-8750 and got %+.4f" % u.metadata["rv"]["rv_measured"])


# ------------------------------------------------------------------------- run
def summary_title(s, stamp, rv, valid):
    hdr = "HE 1002-0755 — %s UT, single exposure" % (
        stamp[len("he1002-0755_"):][:4] + "-" + stamp[len("he1002-0755_"):][4:6] + "-" +
        stamp[len("he1002-0755_"):][6:8] + " " + stamp[len("he1002-0755_"):][9:11] + ":" +
        stamp[len("he1002-0755_"):][11:13])
    tail = "stitched in SMHR, shifted to rest by %+.2f km/s" % rv if valid else \
        "stitched in SMHR, NOT shifted (no valid velocity)"
    return hdr + "\n" + tail


def run(only=None):
    for d in ("", "sessions", "summary_plots"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    sf_default = yaml.safe_load(open(os.path.join(os.path.dirname(smh.__file__),
                                                  "default_session.yaml")))["summary_figure"]
    paths = [p for p in sorted(glob.glob(f"{SESS}/he1002-0755_2*.smh"))
             if not only or only in stamp_of(p)]

    # pass 1: SMHR's velocity in every window, for every exposure
    loaded = []
    for path in paths:
        s = smh.Session.load(path)
        loaded.append((stamp_of(path), s, measure_rvs(s)))
    # the reference must come from ALL exposures, so use every session on disk
    if only:
        ref_rows = [measure_rvs(smh.Session.load(p)) for p in sorted(glob.glob(f"{SESS}/he1002-0755_2*.smh"))]
    else:
        ref_rows = [r for _, _, r in loaded]
    ref = campaign_reference(ref_rows)
    print("campaign reference (adopted window, exposures with signal): %+.2f km/s\n" % ref)

    # pass 2: correct, normalise, stitch, write
    table = []
    for stamp, s, rows in loaded:
        valid, note = rv_verdict(rows, ref)
        ad = [r for r in rows if (r["lo"], r["hi"]) == ADOPTED][0]
        # the last rv_measure call must describe the adopted window; re-run it if
        # `only` reordered nothing, it is already last by construction
        if valid:
            s.rv_correct(ad["rv"])
            rv_used, tag = ad["rv"], ""
        else:
            s.metadata["rv"]["rv_applied"] = 0.0          # deliberately not shifted
            rv_used, tag = 0.0, "_noRV"
        failed = normalise_all(s)
        ns = s.stitch_and_stack()

        base = f"{OUT}/{stamp}{tag}"
        ns.write(base + "_stitched.txt")
        t = Table([ns.dispersion, ns.flux, ns.ivar], names=["wavelength", "norm_flux", "ivar"])
        t.meta.update(OBJECT="HE1002-0755", STAMP=stamp, RVSRC="SMHR rv_measure",
                      RVWIN="%d-%d" % ADOPTED, SHIFTED=bool(valid), TEMPLATE="hd122563")
        if valid:
            t.meta["VGEO"] = float(rv_used)
        t.write(base + "_stitched.fits", overwrite=True)
        s.save(f"{OUT}/sessions/{stamp}{tag}.smh", overwrite=True)

        sf = s.metadata.get("summary_figure") or sf_default
        fig = make_summary_plot(sf, ns)
        fig.suptitle(summary_title(s, stamp, rv_used, valid), fontsize=11)
        for ext in ("png", "pdf"):
            fig.savefig(f"{OUT}/summary_plots/{stamp}{tag}_summary.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)

        g = np.isfinite(ns.flux)
        print("%-30s rv %+9.3f  %s  %5.0f-%5.0f A  finite %.0f%%  cont-failed %d  %s" % (
            stamp + tag, rv_used, "ok  " if valid else "SKIP", ns.dispersion[0], ns.dispersion[-1],
            100 * g.mean(), len(failed), note))
        for r in rows:
            table.append(dict(exposure=stamp, valid=valid, adopted=(r["lo"], r["hi"]) == ADOPTED,
                              win_lo=r["lo"], win_hi=r["hi"], rv_smhr=r["rv"], ccf_peak_width=r["ccf_width"],
                              order=r["order"], order_snr=r["order_snr"], smhr_helio_corr=r["hc"],
                              smhr_bary_corr=r["bc"], note=note if (r["lo"], r["hi"]) == ADOPTED else r["err"]))
    with open(f"{OUT}/rv_smhr_windows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader(); w.writerows(table)
    print("\nwrote %d rows to %s/rv_smhr_windows.csv" % (len(table), OUT))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--only", default=None, help="substring of a session stamp")
    a = ap.parse_args()
    if a.validate:
        validate()
    if a.run:
        run(a.only)
    if not (a.validate or a.run):
        ap.print_help()
