#!/usr/bin/env python
"""Coadd the SMHR-stitched, continuum-normalised exposures into one two-column ASCII spectrum.

SMHR has no exposure-stacking function (Session.stitch_and_stack only merges the orders of one
exposure), and WS3 loads a single ASCII file (wavelength, norm_flux). This combines the outputs of
stitch_sessions.py -- nothing is re-normalised or re-measured here.

Method: every stitched spectrum is already in the rest frame (SMHR applied rv_applied). Each is
interpolated onto one uniform grid, with ivar = 0 wherever the exposure has no data (NaN, ivar = 0,
or further than 2 native pixels from a valid pixel). Pixels more than 5 sigma from the
inverse-variance-weighted mean are rejected once (sigma = formal error plus a 1% floor in quadrature,
because at S/N > 50 exposure-to-exposure differences in line profile and continuum exceed photon noise;
with the formal error alone 13% of pixels were rejected), then the weighted mean is recomputed.
With the floor 8% of pixel-exposures are still rejected, 17% of them redward of 7000 A (telluric
absorption, which moves in the stellar rest frame from exposure to exposure) and 3-5% in the blue/optical
(line-profile differences), so the clip is doing real work in the red; check any line you rely on there.
The formal coadd S/N printed below is optimistic: ivar is interpolated, not conserved.  Same weighting SMHR's own stitch() uses for overlapping orders.

Run in any env with numpy:  python scripts/coadd_smhr.py
Writes data/stitched/he1002-0755_coadd_smhr.txt      (wavelength norm_flux)
       data/stitched/he1002-0755_coadd_smhr_ivar.txt (wavelength norm_flux ivar)
"""
import glob, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/data/stitched"
EXCLUDE = ["_noRV"]          # 2018-03-07: SMHR velocity fails the consistency check (stitch_sessions.rv_verdict)
# Wavelength ranges dropped from one exposure only. 2013-05-31 23:09: the blue arm does not contain the
# star's spectrum (cross-correlation against the coadd peaks at random velocities with r = 0.3-0.6 in every
# blue window, while its red arm matches at 0 km/s with r > 0.9); the header is on target. Found on the
# coadd check sheets (print/2026-10-06). Its red arm stays in.
EXCLUDE_RANGES = {"20130531_2309": [(0.0, 5000.0)]}
DW = 0.03                    # Angstrom; finer than the native ~0.05-0.07 median step
CLIP = 5.0
FLOOR = 0.01                 # fractional systematic floor used only in the clip test, not in the weights

def load(f):
    d = np.loadtxt(f)
    ok = np.isfinite(d[:, 1]) & np.isfinite(d[:, 2]) & (d[:, 2] > 0)
    return d[ok, 0], d[ok, 1], d[ok, 2]

def main():
    files = [f for f in sorted(glob.glob(f"{OUT}/he1002-0755_*_stitched.txt")) if not any(x in f for x in EXCLUDE)]
    spec = []
    for f in files:
        x, y, w = load(f)
        for key, rngs in EXCLUDE_RANGES.items():
            if key in f:
                for a, b in rngs:
                    keep = (x < a) | (x > b); x, y, w = x[keep], y[keep], w[keep]
        spec.append((x, y, w))
    lo = min(s[0].min() for s in spec); hi = max(s[0].max() for s in spec)
    grid = np.arange(np.ceil(lo), np.floor(hi), DW)
    F = np.zeros((len(spec), grid.size)); W = np.zeros_like(F)
    for i, (x, y, w) in enumerate(spec):
        F[i] = np.interp(grid, x, y, left=0, right=0)
        W[i] = np.interp(grid, x, w, left=0, right=0)
        # no data where the grid point is farther than 2 local pixels from a real pixel (gaps between orders)
        j = np.clip(np.searchsorted(x, grid), 1, x.size-1)
        gap = np.maximum(x[j]-x[j-1], 1e-6)
        near = np.minimum(np.abs(grid-x[j]), np.abs(grid-x[j-1]))
        W[i, (near > 2*gap) | (grid < x[0]) | (grid > x[-1])] = 0
    def wmean(W):
        s = W.sum(0)
        return np.where(s > 0, (F*W).sum(0)/np.where(s > 0, s, 1), np.nan), s
    m, s = wmean(W)
    z = np.abs(F-m)/np.sqrt(1.0/np.where(W > 0, W, np.inf) + FLOOR**2)
    nclip = int(((z > CLIP) & (W > 0)).sum())
    W2 = np.where(z > CLIP, 0, W)
    m, s = wmean(W2)
    good = s > 0
    np.savetxt(f"{OUT}/he1002-0755_coadd_smhr.txt", np.c_[grid[good], m[good]], fmt="%.4f %.5f",
               header="wavelength norm_flux", comments="")
    np.savetxt(f"{OUT}/he1002-0755_coadd_smhr_ivar.txt", np.c_[grid[good], m[good], s[good]], fmt="%.4f %.5f %.4e",
               header="wavelength norm_flux ivar", comments="")
    n = (W2 > 0).sum(0)
    print(f"{len(files)} exposures; grid {grid[0]:.0f}-{grid[-1]:.0f} A at {DW} A; {good.sum()} pixels written")
    print(f"clipped {nclip} pixel-exposures ({100*nclip/(W > 0).sum():.3f}%)")
    for k in (1, 2, 5, 8, len(files)):
        print(f"  pixels with >= {k:2d} exposures: {100*(n[good] >= k).mean():5.1f}%")
    for a, b in ((4000, 4100), (5100, 5200), (8450, 8750)):
        sel = good & (grid > a) & (grid < b)
        best = max(np.median(np.sqrt(W[i, sel][W[i, sel] > 0])) if (W[i, sel] > 0).any() else 0 for i in range(len(spec)))
        print(f"  median S/N per pixel {a}-{b}: coadd {np.median(np.sqrt(s[sel])):.1f}, best single exposure {best:.1f}")

if __name__ == "__main__":
    main()
