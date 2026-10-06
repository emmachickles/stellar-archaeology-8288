#!/usr/bin/env python
"""Print sheets that check our own coadd (data/stitched/he1002-0755_coadd_smhr_ivar.txt).

Same layout as print sheets 9-10, but in the REST frame: every input is an SMHR-stitched
spectrum that SMHR already shifted by its rv_applied, so a correct coadd needs every row's lines
to sit on the same rest wavelength, and the coadd (top row) to look like a cleaner version of
each exposure. HD 122563 (rest frame) is overplotted as the reference on every row.

Per row: the exposure's share of the coadd weight in that window (median ivar / sum), and the
DER_SNR S/N measured from the plotted pixels. The coadd row carries the same S/N, so the gain from
stacking can be read straight off the page. 2018-03-07 (excluded: SMHR velocity failed) is drawn
in grey at the bottom for completeness.

Also writes an overview sheet: the whole coadd in strips, with the number of contributing
exposures underneath.

    ~/miniforge3-fresh/envs/wd-periodicity/bin/python scripts/coadd_qa_sheets.py
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from mike_multispec import der_snr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
ST = "data/stitched"
OUT = "print/2026-10-06"
os.makedirs(OUT, exist_ok=True)

REGIONS = [  # (stem, lo, hi, marks, title, ylim)
 ("CaHK",    3925.0, 3980.0, {"Ca II K": 3933.66, "Ca II H": 3968.47}, "Ca II H and K", (-0.05, 1.35)),
 ("Eu4129",  4122.0, 4136.0, {"Eu II 4129": 4129.72}, "Eu II 4129", (0.55, 1.15)),
 ("Mg4167",  4157.0, 4178.0, {"Mg I 4167": 4167.27}, "Mg I 4167", (0.35, 1.25)),
 ("Ba4554",  4547.0, 4562.0, {"Ba II 4554": 4554.03}, "Ba II 4554", (0.25, 1.2)),
 ("Mgb",     5160.0, 5190.0, {"Mg I 5167": 5167.32, "Mg I 5173": 5172.68, "Mg I 5184": 5183.60}, "Mg b triplet", (0.05, 1.2)),
 ("Halpha",  6550.0, 6576.0, {"H$\\alpha$": 6562.80}, "H$\\alpha$", (0.2, 1.25)),
 ("CaT",     8490.0, 8550.0, {"Ca II 8498": 8498.02, "Ca II 8542": 8542.09}, "Ca II triplet (8498, 8542)", (0.2, 1.25)),
]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coadd_smhr import EXCLUDE_RANGES

def load(f):
    d = np.loadtxt(f, skiprows=1) if open(f).readline().startswith("wave") else np.loadtxt(f)
    ok = np.isfinite(d[:, 1]) & (d[:, 2] > 0)
    return d[ok, 0], d[ok, 1], d[ok, 2]

def label(f):
    s = os.path.basename(f).split("_")
    return f"{s[1][:4]}-{s[1][4:6]}-{s[1][6:]} {s[2][:2]}:{s[2][2:]}"

def main():
    files = sorted(glob.glob(f"{ST}/he1002-0755_*_stitched.txt"))
    used = [f for f in files if "_noRV" not in f]
    excl = [f for f in files if "_noRV" in f]
    spec = {f: load(f) for f in files}
    cw, cf, ci = load(f"{ST}/he1002-0755_coadd_smhr_ivar.txt")
    h = fits.open("data/HD122563.fits")[0]; hd = h.header
    wr = hd["CRVAL1"] + hd.get("CD1_1", hd.get("CDELT1"))*(np.arange(hd["NAXIS1"]) + 1 - hd.get("CRPIX1", 1))
    fr = np.asarray(h.data, float)

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 10, "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "legend.frameon": False})
    bbox = dict(fc="white", ec="none", alpha=0.85, pad=1.4)
    summary = []

    for stem, lo, hi, marks, title, ylim in REGIONS:
        rows = [("coadd", cw, cf, ci)] + [(label(f),) + spec[f] for f in used] + [(label(f) + " (excluded)",) + spec[f] for f in excl]
        med_ivar = {}
        for f in used:
            w, y, iv = spec[f]; g = (w > lo) & (w < hi)
            med_ivar[label(f)] = np.median(iv[g]) if g.sum() > 20 else 0.0
        tot = sum(med_ivar.values())
        native = np.median([np.median(np.diff(spec[f][0][(spec[f][0] > lo) & (spec[f][0] < hi)])) for f in used
                            if ((spec[f][0] > lo) & (spec[f][0] < hi)).sum() > 20])
        n = len(rows)
        fig, axes = plt.subplots(n, 1, figsize=(9.0, 0.95*n + 1.2), sharex=True, gridspec_kw=dict(hspace=0.0))
        m = (wr > lo - 2) & (wr < hi + 2)
        snr_rows = {}
        for ax, (tag, w, y, iv) in zip(axes, rows):
            ax.plot(wr[m], fr[m], color="#0072B2", lw=0.7, alpha=0.8)
            g = (w > lo) & (w < hi)
            col, lw = ("#D55E00", 1.1) if tag == "coadd" else (("0.55", 0.8) if "excluded" in tag else ("k", 0.9))
            if g.sum() > 20:
                ax.plot(w[g], y[g], "-", color=col, lw=lw)
                snr = der_snr(y[g]) if tag != "coadd" else np.nan
                formal = float(np.median(np.sqrt(iv[g])))
                snr_rows[tag] = snr
            else:
                snr, formal = np.nan, np.nan
                ax.text(0.5, 0.5, "no coverage", transform=ax.transAxes, ha="center", va="center", fontsize=7, color="0.4")
            for lam in marks.values():
                ax.axvline(lam, color="#0072B2", ls="-", lw=0.6, alpha=0.5)
            ax.set_xlim(lo, hi); ax.set_ylim(*ylim); ax.set_yticks([])
            ax.text(0.006, 0.80, tag, transform=ax.transAxes, fontsize=7.5, family="monospace", va="center",
                    bbox=bbox, zorder=20, color=col if tag == "coadd" else "k")
            if tag == "coadd":
                note = "weighted mean of %d exposures" % len(used)
            elif "excluded" in tag:
                note = "not in the coadd"
            else:
                note = "weight %.0f%%" % (100*med_ivar.get(tag, 0)/tot) if tot > 0 else ""
            dropped = any(k in tag.replace("-", "").replace(":", "").replace(" ", "_") and any(a < hi and b > lo for a, b in r)
                          for k, r in EXCLUDE_RANGES.items())
            if dropped:
                note = "this range excluded from the coadd"
                for l in ax.lines[1:]: l.set_color("0.6")
            if tag == "coadd":
                stxt = "S/N %.0f (formal)" % formal if formal == formal else "S/N --"
            else:
                stxt = ("S/N %.0f measured, %.0f formal" % (snr, formal)) if snr == snr else "S/N --"
            ax.text(0.006, 0.12, note, transform=ax.transAxes, fontsize=7, va="center", bbox=bbox)
            ax.text(0.994, 0.12, stxt, transform=ax.transAxes, fontsize=7, ha="right", va="center", bbox=bbox)
        axes[-1].set_xlabel(r"Rest wavelength, $\lambda$ ($\AA$)")
        for nm, lam in marks.items():
            axes[0].annotate(nm, xy=(lam, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 4),
                             textcoords="offset points", ha="center", fontsize=8, color="#0072B2")
        fig.suptitle(f"{title}: our coadd of HE 1002-0755 and the {len(used)} exposures that make it", y=0.997, fontsize=11)
        fig.text(0.5, 0.952, "Rest frame (SMHR-stitched, SMHR velocity applied). HD 122563 (blue) is at rest; vertical "
                 "lines mark rest wavelengths.\nCoadd in orange. Weight = the exposure's share of the inverse "
                 "variance in this window; S/N: measured = DER_SNR on the plotted pixels;\nformal = median sqrt(ivar) from the pipeline noise. The coadd is on an oversampled grid, so only its formal S/N is shown.", ha="center", fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        for ext in ("png", "pdf"):
            fig.savefig(f"{OUT}/coadd_{stem}.{ext}", dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"{title:28s} done")

    # ---- overview: whole coadd in strips + coverage
    grid_n = np.zeros_like(cw)
    for f in used:
        w, _, _ = spec[f]
        j = np.clip(np.searchsorted(w, cw), 1, len(w) - 1)
        near = np.minimum(abs(cw - w[j]), abs(cw - w[j-1])) < 0.1
        grid_n += near & (cw >= w[0]) & (cw <= w[-1])
    edges = np.linspace(cw[0], cw[-1], 11)
    fig, axes = plt.subplots(10, 1, figsize=(9.0, 12.5))
    for ax, a, b in zip(axes, edges[:-1], edges[1:]):
        g = (cw >= a) & (cw < b)
        ax.plot(cw[g], cf[g], "k-", lw=0.35)
        ax.set_xlim(a, b); ax.set_ylim(-0.05, 1.4); ax.set_yticks([0, 0.5, 1.0])
        ax.axhline(1, color="#0072B2", lw=0.5, alpha=0.6)
        ax2 = ax.twinx(); ax2.fill_between(cw[g], 0, grid_n[g], color="#E69F00", alpha=0.25, lw=0, step="mid")
        ax2.set_ylim(0, 4*len(used)); ax2.set_yticks([0, len(used)]); ax2.tick_params(labelsize=6, colors="#B07800")
        for lo_, hi_ in ((6860, 6960), (7590, 7700), (8100, 8400), (8900, 9400)):
            if hi_ > a and lo_ < b: ax.axvspan(max(lo_, a), min(hi_, b), color="0.85", lw=0, zorder=0)
    axes[-1].set_xlabel(r"Rest wavelength ($\AA$)")
    fig.suptitle("Our coadd of HE 1002-0755, full range (normalised flux; orange = number of exposures contributing,\n"
                 "right axis; grey bands = strong telluric regions)", fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/coadd_overview.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
