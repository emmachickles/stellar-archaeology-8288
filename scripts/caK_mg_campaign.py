"""Recipe from the course instructions: normalise only the orders holding
Ca II K and Mg I 4167, coadd those, and plot velocity against Julian date.

Run:  conda activate wd-periodicity && python scripts/caK_mg_campaign.py
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from mike_multispec import read_multispec, normalise_order, despike, der_snr, C_KMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
RES = "worksheets/ws2/part2_observation_details/results"
PLOTS = "worksheets/ws2/part2_observation_details/plots"

# the two lines suggested in class, plus Ca II H which rides in the same orders
TARGETS = {"Ca II K": 3933.66, "Ca II H": 3968.47, "Mg I 4167": 4167.27}
HALF = 9.0

def pick_order(orders, obs, half=HALF):
    """Highest-S/N order that fully contains the line plus its window."""
    cands = [o for o in orders if o[0][0] < obs - half - 4 and o[0][-1] > obs + half + 4]
    return max(cands, key=lambda o: np.nanmedian(o[2])) if cands else None

def main():
    rv = json.load(open(f"{RES}/rv_caK_mg.json"))
    # ---- velocities -> heliocentric, per line
    rows = []
    for r in rv:
        row = dict(epoch=r["epoch"], jd=r["jd"], helcorr=r["helcorr"])
        ref = r.get("Mg I 4167")
        for nm in TARGETS:
            v = r.get(nm)
            # a line that disagrees with Mg I 4167 by >20 km/s has not been measured
            if v is not None and v == v and ref is not None and ref == ref and abs(v - ref) > 20:
                v = np.nan
            row[nm] = (v + r["helcorr"]) if (v is not None and v == v) else np.nan
        rows.append(row)
    jd = np.array([r["jd"] for r in rows])
    out = {nm: np.array([r[nm] for r in rows]) for nm in TARGETS}

    print("%-12s %10s %10s %10s" % ("epoch", "Ca II K", "Ca II H", "Mg I 4167"))
    for r in rows:
        f = lambda k: ("%+10.2f" % r[k]) if r[k] == r[k] else "       ---"
        print("%-12s %s %s %s" % (r["epoch"], f("Ca II K"), f("Ca II H"), f("Mg I 4167")))
    for nm, v in out.items():
        g = v[np.isfinite(v)]
        print("  %-10s mean %+7.2f  scatter %.2f  (n=%d)" % (nm, g.mean(), g.std(ddof=1), len(g)))

    # ---- figure: v_helio vs JD
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 10,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
        "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True,
        "legend.frameon": False, "axes.linewidth": 0.9})
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    style = {"Ca II K": ("o", "#000000", 6), "Mg I 4167": ("s", "#0072B2", 5),
             "Ca II H": ("^", "#D55E00", 5)}
    for nm in ["Ca II K", "Mg I 4167", "Ca II H"]:
        m, c, ms = style[nm]
        g = np.isfinite(out[nm])
        ax.plot(jd[g], out[nm][g], m, color=c, ms=ms,
                mfc="none" if nm != "Ca II K" else c, label=nm)
    allv = np.concatenate([out[n][np.isfinite(out[n])] for n in TARGETS])
    ax.axhline(allv.mean(), color="0.6", ls=":", lw=1)
    ax.set_xlabel("Julian date"); ax.set_ylabel(r"$v_{\rm helio}$  (km s$^{-1}$)")
    ax.set_title("HE 1002-0755: velocity over a 9.1 yr baseline", fontsize=11)
    ax.legend(loc="lower left", ncol=3, fontsize=8.5)
    ax.grid(alpha=0.18, lw=0.6)
    for ext in ("png", "pdf"):
        fig.savefig(f"{PLOTS}/rv_vs_jd_caK_mg.{ext}", dpi=200, bbox_inches="tight")
    print(f"\nwrote {PLOTS}/rv_vs_jd_caK_mg.{{png,pdf}}")

    # ---- focused coadd: only the orders holding those lines
    # This is the point of the recipe -- the whole 70-order stitch is
    # not needed if the abundance work only uses Ca II K and Mg I 4167.
    D = "data/he1002-0755_epochs"
    PAIRS = {
     "2009-02-05": ("he1002-0755ablue_multi_ut090205_07slit.fits", "he1002-0755ared_multi_ut090205_07slit.fits"),
     "2009-02-06": ("he1002-0755bblue_multi_ut090205_07slit.fits", "he1002-0755bred_multi_ut090205_07slit.fits"),
     "2009-02-07": ("he1002-0755cblue_multi_ut090205_07slit.fits", "he1002-0755cred_multi_ut090205_07slit.fits"),
     "2010-05-08": ("he1002-0755_2010_blue_multi.fits", "he1002-0755_2010_red_multi.fits"),
     "2011-03-10": ("he1002-0755blue_multi_ut110310_07slit.fits", "he1002-0755red_multi_ut110310_07slit.fits"),
     "2013-05-30a": ("he1002-0755blue_multi_ut130530_07slit.fits", "he1002-0755red_multi_ut130530_07slit.fits"),
     "2013-05-30b": ("he1002-0755_2013_may30_blue_multi.fits", "he1002-0755_2013_may30_red_multi.fits"),
     "2013-05-31": ("he1002-0755_2013_may31_blue_multi.fits", "he1002-0755_2013_may31_red_multi.fits"),
     "2014-03-12": ("he1002-0755_2014_blue_multi.fits", "he1002-0755_2014_red_multi.fits"),
     "2017-06-05": ("he1002-0755_2017_blue_multi.fits", "he1002-0755_2017_red_multi.fits")}
    def path(n):
        q = os.path.join(D, n)
        return q if os.path.exists(q) else os.path.join("data", n)
    vgeo = {r["epoch"]: r for r in rv}

    REGIONS = {"CaHK": (3925.0, 3978.0), "Mg4167": (4158.0, 4177.0)}
    coadds = {}
    for rname, (rlo, rhi) in REGIONS.items():
        stack = []
        for tag, (b, rr) in PAIRS.items():
            v = vgeo[tag].get("Mg I 4167")          # one velocity per epoch, the reliable line
            if v is None or v != v:
                continue
            ords = []
            for f in (path(b), path(rr)):
                o, _ = read_multispec(f); ords += o
            mid = 0.5*(rlo + rhi)*(1 + v/C_KMS)
            o = pick_order(ords, mid, half=0.5*(rhi - rlo))
            if o is None:
                continue
            w, fl, s = o
            nf, _ = normalise_order(w, despike(fl),
                                    mask=[(3930*(1+v/C_KMS), 3975*(1+v/C_KMS))] if rname == "CaHK" else None,
                                    deg=3)
            wr = w/(1 + v/C_KMS)
            g = np.isfinite(nf) & (wr > rlo - 2) & (wr < rhi + 2)
            if g.sum() < 50:
                continue
            stack.append((wr[g], nf[g], max(np.nanmedian(s), 1.0)))
        if not stack:
            continue
        grid = np.arange(rlo, rhi, 0.03)
        num = np.zeros_like(grid); den = np.zeros_like(grid)
        for wr, nf, wt in stack:
            fi = np.interp(grid, wr, nf, left=np.nan, right=np.nan)
            ok = np.isfinite(fi)
            num[ok] += fi[ok]*wt**2; den[ok] += wt**2
        with np.errstate(invalid="ignore", divide="ignore"):
            fc = num/den
        k = den > 0
        coadds[rname] = (grid[k], fc[k], len(stack))
        print("  %-7s coadd from %d epochs: %.1f-%.1f A, DER_SNR %.0f"
              % (rname, len(stack), grid[k][0], grid[k][-1], der_snr(fc[k])))
        t = Table([grid[k], fc[k]], names=["wavelength", "norm_flux"])
        t.meta["NEPOCH"] = len(stack)
        t.write("data/he1002-0755_coadd_%s.fits" % rname, overwrite=True)

    # ---- figure: the two focused coadds
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.0))
    ref = {}
    for nm in ("hd122563", "he1523-0901"):
        h = fits.open("data/%s.fits" % ("HD122563" if nm == "hd122563" else nm))[0]
        hd = h.header; n = hd["NAXIS1"]; cd = hd.get("CD1_1", hd.get("CDELT1"))
        ref[nm] = (hd["CRVAL1"] + cd*(np.arange(n) + 1 - hd.get("CRPIX1", 1)),
                   np.asarray(h.data, float))
    for ax, (rname, lab) in zip(axes, [("CaHK", "Ca II H and K"), ("Mg4167", "Mg I 4167")]):
        if rname not in coadds:
            continue
        w, f, ne = coadds[rname]
        for nm, col in [("hd122563", "#0072B2"), ("he1523-0901", "#D55E00")]:
            wr, fr = ref[nm]
            m = (wr > w[0]) & (wr < w[-1])
            ax.plot(wr[m], fr[m], color=col, lw=0.8, alpha=0.85, label=nm)
        ax.plot(w, f, "k-", lw=1.4, label="HE 1002-0755 (%d exposures)" % ne)
        ax.set_xlim(w[0], w[-1]); ax.set_ylim(-0.05, 1.3)
        ax.set_ylabel("Normalised flux")
        ax.text(0.015, 0.06, lab, transform=ax.transAxes, fontsize=10, weight="bold")
        ax.legend(fontsize=8, loc="lower right", ncol=3)
        ax.grid(alpha=0.18, lw=0.6)
    axes[-1].set_xlabel(r"Rest wavelength, $\lambda$ ($\AA$)")
    fig.suptitle("HE 1002-0755: coadded Ca II H, K and Mg I 4167 regions", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig("worksheets/ws2/part3_summary_plot/plots/focused_coadds.%s" % ext,
                    dpi=200, bbox_inches="tight")
    print("wrote worksheets/ws2/part3_summary_plot/plots/focused_coadds.{png,pdf}")
    return rows

if __name__ == "__main__":
    main()
