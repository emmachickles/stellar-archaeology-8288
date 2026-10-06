"""Radial-velocity curve of HE 1002-0755 from results/rv_lines_per_exposure.json.

Heliocentric velocities -- the quantity an RV curve is made of -- with per-exposure
uncertainties from the line-to-line scatter, and every exposure marked.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
RES = "worksheets/ws2/part2_observation_details/results"
PLOTS = "worksheets/ws2/part2_observation_details/plots"
LINES = ["Ca II K", "Ca II H", "Mg I 4167"]

def main():
    rec = json.load(open(f"{RES}/rv_lines_per_exposure.json"))
    use = [r for r in rec if r["v_helio"] is not None]
    jd = np.array([r["jd"] for r in use])
    v = np.array([r["v_helio"] for r in use])
    e = np.array([r["v_helio_err"] for r in use])
    lab = [r["exposure"] for r in use]

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 10,
        "axes.linewidth": 0.9, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True,
        "ytick.minor.visible": True, "legend.frameon": False})

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True,
                                  gridspec_kw=dict(hspace=0.07, height_ratios=[2.6, 1.0]))

    # Every exposure gets a marker line, but at a 9-year x-scale the three
    # February 2009 exposures (and the three in May 2013) fall on the same pixel,
    # so labels are placed per observing run, not per exposure.
    for x in jd:
        for a in (ax, axr):
            a.axvline(x, color="0.82", lw=0.6, zorder=0)
    runs, cur = [], [0]
    for i in range(1, len(jd)):
        (cur.append(i) if jd[i] - jd[cur[-1]] < 30 else (runs.append(cur), cur := [i]))
    runs.append(cur)
    for k, idx in enumerate(runs):
        xc = float(np.mean(jd[idx]))
        first = lab[idx[0]][:10]
        txt = first if len(idx) == 1 else "%s  x%d" % (first, len(idx))
        ax.annotate(txt, xy=(xc, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 5), textcoords="offset points", rotation=90,
                    ha="center", va="bottom", fontsize=7, family="monospace",
                    color="0.30")

    w = 1.0/e**2
    wm = float(np.sum(v*w)/np.sum(w))
    wm_err = float(np.sqrt(1.0/np.sum(w)))
    ax.axhline(wm, color="#0072B2", lw=1.1, ls="--",
               label=r"weighted mean $%+.2f \pm %.2f$ km s$^{-1}$" % (wm, wm_err))
    ax.fill_between([jd.min() - 200, jd.max() + 200], wm - wm_err, wm + wm_err,
                    color="#0072B2", alpha=0.15, lw=0)

    # the individual lines, faint, behind the mean
    for nm, mk, col in zip(LINES, ["v", "^", "s"], ["#999999", "#bbbbbb", "#777777"]):
        xs = [r["jd"] for r in use if r.get(nm) is not None]
        ys = [r[nm] for r in use if r.get(nm) is not None]
        ax.plot(xs, ys, mk, ms=3.2, mfc="none", color=col, lw=0, label=nm, zorder=2)

    ax.errorbar(jd, v, yerr=e, fmt="o", ms=5.5, color="black", capsize=2.5,
                lw=1.1, zorder=5, label="mean of available lines")
    ax.set_ylabel(r"$v_{\rm helio}$  (km s$^{-1}$)")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(alpha=0.15, lw=0.6)

    res = v - wm
    axr.errorbar(jd, res, yerr=e, fmt="o", ms=5.0, color="black", capsize=2.5, lw=1.1)
    axr.axhline(0, color="#0072B2", lw=1.0, ls="--")
    axr.set_ylabel("O$-$C (km s$^{-1}$)")
    axr.set_xlabel("Julian date")
    axr.grid(alpha=0.15, lw=0.6)
    axr.set_xlim(jd.min() - 150, jd.max() + 150)

    chi2 = float(np.sum((res/e)**2)); dof = len(v) - 1
    axr.text(0.985, 0.08, r"$\chi^2/\nu = %.1f/%d = %.1f$" % (chi2, dof, chi2/dof),
             transform=axr.transAxes, ha="right", fontsize=8)

    fig.suptitle("HE 1002-0755: heliocentric radial velocity over a 9.1 yr baseline",
                 y=1.045, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    for ext in ("png", "pdf"):
        fig.savefig(f"{PLOTS}/rv_curve_HE1002-0755.{ext}", dpi=200, bbox_inches="tight")
    print("wrote %s/rv_curve_HE1002-0755.{png,pdf}" % PLOTS)
    print("weighted mean %+.2f +/- %.2f   chi2/dof = %.1f/%d = %.1f   scatter %.2f"
          % (wm, wm_err, chi2, dof, chi2/dof, v.std(ddof=1)))

if __name__ == "__main__":
    main()
