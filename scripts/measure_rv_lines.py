"""Per-exposure radial velocities of HE 1002-0755 from three narrow metal lines.

Each line is measured on the single highest-S/N echelle order that contains it,
normalised on its own -- no merging, so an exposure cannot be contaminated by
another. Writes results/rv_lines_per_exposure.json.

The reported velocity per exposure is the mean over the lines that were
measurable, and its uncertainty is the standard error of that mean. That is a
line-to-line systematic, which dominates: the CCF formal error is ~0.1 km/s and
badly understates the truth.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from PyAstronomy import pyasl
from mike_multispec import read_multispec, normalise_order, despike, der_snr, C_KMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
D = "data/he1002-0755_epochs"
RES = "worksheets/ws2/part2_observation_details/results"

EXPOSURES = [
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
LINES = {"Ca II K": 3933.66, "Ca II H": 3968.47, "Mg I 4167": 4167.27}
HALF = 9.0
VNOM = 225.0          # only used to place the window and pick the order
VMIN, VMAX = 80.0, 420.0
LCO = EarthLocation.from_geodetic(lon=-70.69242*u.deg, lat=-29.01423*u.deg, height=2400*u.m)

def path(n):
    p = os.path.join(D, n)
    return p if os.path.exists(p) else os.path.join("data", n)

def load1d(p):
    h = fits.open("data/" + p)[0]; hd = h.header; n = hd["NAXIS1"]
    cd = hd.get("CD1_1", hd.get("CDELT1"))
    return hd["CRVAL1"] + cd*(np.arange(n) + 1 - hd.get("CRPIX1", 1)), np.asarray(h.data, float)

def ccf(w, f, wt, ft, lo, hi, pad=20):
    m = (w > lo) & (w < hi); mt = (wt > lo - pad) & (wt < hi + pad)
    if m.sum() < 30 or mt.sum() < 30:
        return np.nan
    r, c = pyasl.crosscorrRV(w[m], f[m] - np.nanmedian(f[m]),
                             wt[mt], ft[mt] - np.nanmedian(ft[mt]),
                             VMIN, VMAX, 0.1, skipedge=10)
    pk = float(r[np.argmax(c)])
    # a peak on the scan boundary is not a measurement
    return np.nan if (pk - VMIN < 5 or VMAX - pk < 5) else pk

def main():
    wt, ft = load1d("HD122563.fits")
    out = []
    for tag, b, r in EXPOSURES:
        ords = []
        hdr = None
        for f in (path(b), path(r)):
            o, h = read_multispec(f)
            if hdr is None:
                hdr = h
            ords += o
        t = Time("%sT%s" % (hdr["UT-DATE"], hdr["UT-START"]), scale="utc", location=LCO)
        sky = SkyCoord(ra=hdr["RA-D"]*u.deg, dec=hdr["DEC-D"]*u.deg)
        hc = float(sky.radial_velocity_correction("heliocentric", obstime=t).to(u.km/u.s).value)
        rec = dict(exposure=tag, iso=t.isot, jd=float(t.jd), helcorr=hc,
                   exptime=float(hdr["EXPTIME"]), airmass=hdr.get("AIRMASS"))
        for nm, lam in LINES.items():
            obs = lam*(1 + VNOM/C_KMS)
            cands = [o for o in ords if o[0][0] < obs - HALF - 4 and o[0][-1] > obs + HALF + 4]
            if not cands:
                rec[nm] = None; continue
            w, fl, s = max(cands, key=lambda o: np.nanmedian(o[2]))
            nf, _ = normalise_order(w, despike(fl), mask=[(obs - HALF - 4, obs + HALF + 4)], deg=3)
            v = ccf(w, nf, wt, ft, obs - HALF, obs + HALF)
            rec[nm] = None if v != v else v + hc      # heliocentric, per line
            if nm == "Mg I 4167":
                g = (w > obs - 12) & (w < obs + 12) & np.isfinite(nf)
                rec["snr_mg"] = float(der_snr(nf[g])) if g.sum() else None
        out.append(rec)

    # --- reject bad lines against the CAMPAIGN median, not the per-exposure one.
    # With two values and one of them wrong, a per-exposure median sits between
    # them and throws away both (2013-05-31 lost every line that way).
    seed = []
    for r in out:
        v = sorted(x for x in (r.get(nm) for nm in LINES) if x is not None)
        # an exposure where two lines agree closely is trustworthy enough to seed
        for i in range(len(v) - 1):
            if v[i+1] - v[i] < 5:
                seed += [v[i], v[i+1]]
    ref = float(np.median(seed)) if seed else VNOM
    print("campaign reference velocity for outlier rejection: %+.2f km/s\n" % ref)

    for rec in out:
        # a frame with no signal cannot yield a velocity, however clean the CCF looks
        if rec.get("snr_mg") is not None and rec["snr_mg"] < 3.0:
            for nm in LINES:
                rec[nm] = None
            rec["rejected"] = "no signal (S/N < 3)"
        for nm in LINES:
            if rec.get(nm) is not None and abs(rec[nm] - ref) > 20:
                rec[nm] = None
        vals = [rec[nm] for nm in LINES if rec.get(nm) is not None]
        rec["n_lines"] = len(vals)
        rec["v_helio"] = float(np.mean(vals)) if vals else None
        rec["v_helio_err"] = (float(np.std(vals, ddof=1)/np.sqrt(len(vals)))
                              if len(vals) > 1 else None)

    # exposures with a single usable line get the campaign-wide line-to-line
    # scatter rather than no uncertainty at all
    multi = [r["v_helio_err"] for r in out if r["v_helio_err"] is not None]
    fallback = float(np.median(multi)) if multi else None
    for r in out:
        if r["v_helio"] is not None and r["v_helio_err"] is None:
            r["v_helio_err"] = fallback
            r["err_is_fallback"] = True

    print("%-17s %4s %10s %8s   %s" % ("exposure", "n", "v_helio", "err", "per line"))
    for r in out:
        per = "  ".join("%s %.2f" % (k.split()[-1], r[k]) for k in LINES if r.get(k) is not None)
        if r["v_helio"] is None:
            print("%-17s %4d %10s %8s   %s" % (r["exposure"], 0, "--", "--", "no usable line"))
        else:
            print("%-17s %4d %+10.2f %8.2f%s   %s" % (r["exposure"], r["n_lines"], r["v_helio"],
                  r["v_helio_err"], "*" if r.get("err_is_fallback") else " ", per))
    print("\n* uncertainty is the campaign median (%.2f km/s); only one line measurable" % fallback)
    v = np.array([r["v_helio"] for r in out if r["v_helio"] is not None])
    print("mean %+.2f, scatter %.2f km/s over %d exposures" % (v.mean(), v.std(ddof=1), len(v)))
    json.dump(out, open(f"{RES}/rv_lines_per_exposure.json", "w"), indent=2)
    print("wrote %s/rv_lines_per_exposure.json" % RES)

if __name__ == "__main__":
    main()
