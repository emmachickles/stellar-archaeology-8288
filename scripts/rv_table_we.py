#!/usr/bin/env python
"""Per-exposure SMHR velocities, with the barycentric correction recomputed by
Wright & Eastman (2014) via barycorrpy alongside the one SMHR reports.

Run in the wd-periodicity env (barycorrpy lives there, not in smhr-py3):
    conda activate wd-periodicity && python scripts/rv_table_we.py

Reads  data/stitched/rv_smhr_windows.csv  (written by stitch_sessions.py)
Writes data/stitched/rv_smhr_summary.csv

Why: for the 2009-02-05 exposure SMHR reports a barycentric correction of
+10.231 km/s, while barycorrpy at mid-exposure gives +9.966 and astropy agrees
with that to 1 m/s. SMHR's stored correction is therefore 0.27 km/s high there.
The measured velocity (the cross-correlation) is unaffected; only the step from
geocentric to barycentric is. This script shows how large the gap is for every
exposure so it can be judged rather than assumed.
"""
import os, glob, csv
import numpy as np
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
import astropy.units as u
from barycorrpy import get_BC_vel

ROOT = os.environ.get("COURSE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{ROOT}/data/stitched"


def header_for(stamp):
    """FITS header of the exposure named he1002-0755_<YYYYMMDD>_<HHMM>[_noRV]."""
    core = stamp.replace("_noRV", "")
    d, t = core.split("_")[1], core.split("_")[2]
    want_date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    want_min = f"{t[:2]}:{t[2:]}"
    for f in sorted(glob.glob(f"{ROOT}/data/he1002-0755*.fits") +
                    glob.glob(f"{ROOT}/data/he1002-0755_epochs/*.fits")):
        if "_rest" in f or "_coadd" in f:
            continue
        h = fits.getheader(f)
        if h.get("UT-DATE") == want_date and str(h.get("UT-START", ""))[:5] == want_min \
                and "Red" in str(h.get("INSTRUME", "")):
            return h
    raise FileNotFoundError(stamp)


def we_correction(h):
    """Barycentric correction in km/s, at the exposure midpoint."""
    sc = SkyCoord(h["RA"], h["DEC"], unit=(u.hourangle, u.deg))
    t0 = Time(f"{h['UT-DATE']}T{h['UT-START']}", scale="utc")
    tm = t0 + TimeDelta(float(h["EXPTIME"]) / 2.0, format="sec")
    r = get_BC_vel(JDUTC=tm.jd, ra=sc.ra.deg, dec=sc.dec.deg, lat=h["SITELAT"],
                   longi=h["SITELONG"], alt=h["SITEALT"], epoch=2451545.0,
                   pmra=0, pmdec=0, px=0, rv=0)
    return float(r[0][0]) / 1e3


def main():
    rows = list(csv.DictReader(open(f"{OUT}/rv_smhr_windows.csv")))
    by = {}
    for r in rows:
        by.setdefault(r["exposure"], []).append(r)

    label = {(8450., 8750.): "rv_CaTriplet", (5100., 5200.): "rv_MgB", (6510., 6610.): "rv_Halpha",
             (4810., 4910.): "rv_Hbeta", (4290., 4390.): "rv_4290"}
    out = []
    for stamp, rs in by.items():
        h = header_for(stamp)
        rec = dict(exposure=stamp, valid=rs[0]["valid"])
        for r in rs:
            rec[label[(float(r["win_lo"]), float(r["win_hi"]))]] = float(r["rv_smhr"])
        adopted = [r for r in rs if r["adopted"] == "True"][0]
        smhr_b = float(adopted["smhr_bary_corr"])
        we_b = we_correction(h)
        rec.update(smhr_bary_corr=smhr_b, we_bary_corr=we_b, smhr_minus_we=smhr_b - we_b)
        v = float(adopted["rv_smhr"])
        rec.update(v_geo_MgB=v, v_bary_we=v + we_b, v_bary_smhr=v + smhr_b)
        out.append(rec)

    cols = ["exposure", "valid", "rv_MgB", "rv_CaTriplet", "rv_4290", "rv_Halpha", "rv_Hbeta",
            "smhr_bary_corr", "we_bary_corr", "smhr_minus_we", "v_bary_we", "v_bary_smhr"]
    with open(f"{OUT}/rv_smhr_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in out:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

    print("%-26s %-5s %9s %9s %9s | %8s %8s %6s | %9s" % (
        "exposure", "valid", "MgB", "Ca trip.", "4290-4390", "SMHR bc", "W&E bc", "diff", "v_bary(W&E)"))
    for r in out:
        f = lambda k: ("%9.2f" % r[k]) if k in r and np.isfinite(r[k]) else "      n/a"
        print("%-26s %-5s %s %s %s | %8.3f %8.3f %6.3f | %9.2f" % (
            r["exposure"], r["valid"], f("rv_MgB"), f("rv_CaTriplet"), f("rv_4290"),
            r["smhr_bary_corr"], r["we_bary_corr"], r["smhr_minus_we"], r["v_bary_we"]))
    d = np.array([r["smhr_minus_we"] for r in out])
    print("\nSMHR minus Wright&Eastman: mean %+.3f, min %+.3f, max %+.3f km/s" % (d.mean(), d.min(), d.max()))


if __name__ == "__main__":
    main()
