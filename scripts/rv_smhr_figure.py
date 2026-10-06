#!/usr/bin/env python
"""RV curve of HE 1002-0755 from the SMHR measurements (data/stitched/rv_smhr_summary.csv).

SMHR cross-correlation against HD 122563 over 8450-8750 A (Ca II triplet), barycentric correction
from barycorrpy at mid-exposure (rv_table_we.py). SMHR gives no usable velocity uncertainty, so the
points carry none; the grey band marks the 0.2-2.2 km/s spread between SMHR windows (Ca triplet vs
Mg b) as a guide to the systematic level. Optional comparison velocities (RV_COMPARE_CSV) are overplotted.

    ~/miniforge3-fresh/envs/wd-periodicity/bin/python scripts/rv_smhr_figure.py
"""
import csv, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.time import Time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = "worksheets/ws2/part2_observation_details/plots"
rows = [r for r in csv.DictReader(open("data/stitched/rv_smhr_summary.csv")) if r["valid"] == "True"]
log = json.load(open("worksheets/ws2/part2_observation_details/results/observing_log.json"))
mid = {}
for r in log:
    if "Red" in r["arm"]:
        key = r["ut"][:16].replace("-", "").replace("T", "_").replace(":", "")
        mid[key] = r["jd"] + r["exptime"]/2/86400.
jd = np.array([mid[r["exposure"].split("he1002-0755_")[1][:13]] for r in rows])
v = np.array([float(r["v_bary_we"]) for r in rows])
# Optional comparison velocities: a two-column CSV (exposure stamp prefix, velocity), e.g. from a
# collaborator's co-add notes. Not distributed with this code.
COMPARE = os.environ.get("RV_COMPARE_CSV", "data/rv_compare.csv")
compare = dict((k, float(v)) for k, v in csv.reader(open(COMPARE))) if os.path.exists(COMPARE) else {}
cj, cv = [], []
for r, j in zip(rows, jd):
    stamp = r["exposure"].split("_", 1)[1]
    for k, val in compare.items():
        if stamp.startswith(k):
            cj.append(j); cv.append(val)
yr = Time(jd, format="jd").decimalyear

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 10, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "legend.frameon": False})
fig, ax = plt.subplots(figsize=(7.0, 3.4))
ax.axhspan(np.mean(v) - 1.1, np.mean(v) + 1.1, color="0.9", lw=0, label="$\\pm$1.1 km s$^{-1}$ (half the window-to-window spread)")
ax.plot(yr, v, "o", color="k", ms=5, label="SMHR, Ca II triplet, barycentric (this work)")
ax.plot(Time(np.array(cj), format="jd").decimalyear, cv, "s", mfc="none", mec="#D55E00", ms=6,
        label="comparison velocities")
ax.axhline(226.68, color="#0072B2", lw=0.8, ls="--", label="Gaia DR3 mean, $+226.68 \\pm 3.71$")
ax.set_xlabel("year"); ax.set_ylabel("$v_r$ (km s$^{-1}$)"); ax.set_ylim(218.5, 229.5)
ax.legend(fontsize=7, loc="lower left", ncol=2)
fig.tight_layout()
for e in ("pdf", "png"): fig.savefig(f"{OUT}/rv_smhr_HE1002-0755.{e}", dpi=200)
print(len(v), "SMHR points;", len(cv), "co-add-note points; mean", round(v.mean(), 2), "range", round(v.min(), 2), round(v.max(), 2))
