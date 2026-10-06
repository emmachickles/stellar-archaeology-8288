"""Create one SMHR session per exposure, each holding its blue+red pair.

Equivalent to shift-clicking the two arms in the GUI's File > New Session, but
done for every exposure at once.

Run in the smhr-py3 env:
    conda activate smhr-py3 && python scripts/make_smh_sessions.py [--overwrite]

Pairing is derived from the FITS headers, not from a hardcoded list, so it
cannot drift from the data:
  * exposures are grouped by UT-START to the MINUTE -- the 2017 blue and red
    differ by one second and would otherwise count as two exposures;
  * where the same arm of the same exposure exists more than once (the 2010 and
    2013 files arrived under two naming schemes), the file with the most
    extracted orders wins, then the higher median S/N.

Naming:  he1002-0755_<YYYYMMDD>_<HHMM>.smh   -- UT, sortable, one per exposure.
"""
import os, sys, glob, argparse
import numpy as np
from astropy.io import fits
import smh
import yaml

ROOT = os.environ.get("COURSE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEARCH = [f"{ROOT}/data/he1002-0755*.fits", f"{ROOT}/data/he1002-0755_epochs/*.fits"]
OUT = f"{ROOT}/data/smh_sessions"

def survey():
    seen = {}
    for pat in SEARCH:
        for f in sorted(glob.glob(pat)):
            if "_rest" in f or "_coadd" in f:
                continue
            hd = fits.open(f)[0]
            h = hd.header
            # OBJECT is spelled three ways across the archive:
            # "HE1002-0755", "HE 1002-0755" and "he1002-0755"
            obj = str(h.get("OBJECT", "")).lower().replace(" ", "").replace("-", "")
            if obj != "he10020755":
                continue
            arm = "blue" if "Blue" in str(h.get("INSTRUME", "")) else "red"
            key = "%sT%s" % (h["UT-DATE"], h["UT-START"][:5])      # to the minute
            norders = hd.data.shape[1]
            snr = float(np.nanmedian(hd.data[3])) if hd.data.shape[0] > 3 else 0.0
            seen.setdefault(key, {}).setdefault(arm, []).append((norders, snr, f))
    return seen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    seen = survey()
    print("%d exposures\n" % len(seen))
    for key in sorted(seen):
        arms = seen[key]
        chosen, dropped = [], []
        for arm in ("blue", "red"):
            if arm not in arms:
                continue
            cands = sorted(arms[arm], key=lambda t: (t[0], t[1]), reverse=True)
            chosen.append(cands[0][2])
            dropped += [c[2] for c in cands[1:]]
        stem = "he1002-0755_%s_%s" % (key[:10].replace("-", ""), key[11:].replace(":", ""))
        path = os.path.join(OUT, stem + ".smh")
        if os.path.exists(path) and not args.overwrite:
            print("%-34s exists, skipped" % (stem + ".smh")); continue
        if len(chosen) < 2:
            print("%-34s only %d arm(s) -- skipped" % (stem + ".smh", len(chosen))); continue
        s = smh.Session(chosen)
        # A GUI-made session is seeded from default_session.yaml; Session()
        # leaves metadata["rv"] empty, and the RV tab then dies with
        # KeyError: 'normalization' the moment you change the wavelength region.
        defaults = yaml.safe_load(open(os.path.join(os.path.dirname(smh.__file__),
                                                    "default_session.yaml")))
        for block in ("rv", "normalization", "isotopes", "summary_figure"):
            if block in defaults:
                cur = s.metadata.setdefault(block, {})
                if isinstance(cur, dict):
                    for k, v in defaults[block].items():
                        cur.setdefault(k, v)
        s.save(path, overwrite=True)
        print("%-34s %2d orders  %5.1f MB" % (stem + ".smh", len(s.input_spectra),
                                              os.path.getsize(path)/1e6))
        for c in chosen:
            print("        + %s" % os.path.basename(c))
        for d in dropped:
            print("        - %s  (duplicate arm, not used)" % os.path.basename(d))
    print("\nwrote to %s" % OUT)

if __name__ == "__main__":
    main()
