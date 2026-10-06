# 8.288 stellar archaeology: analysis and plotting code

Code from MIT 8.288 (Observational Stellar Archaeology, Fall 2026) for Magellan/MIKE echelle spectra:
order handling, SMHR (Spectroscopy Made Harder) sessions, radial velocities, co-adding, QA plots, and
the Worksheet 1–4 notebooks (stellar parameters, Fe I diagnostics, abundance tables).

**No course spectra are included.** Notebook outputs are cleared, except where they come only from
public data (Gaia, APASS/UCAC4, 2MASS, IRSA dust maps): the WS4 stellar-parameters notebook and the
figures in `figures/`. To use the code, put your own spectra in a
`data/` directory next to `scripts/`, or set `COURSE_ROOT` to the directory that contains `data/`.
Star names and file names in the scripts are for HE 1002-0755 and should be changed for your star.

| Path | What |
|---|---|
| `scripts/mike_multispec.py` | read IRAF multispec MIKE files (all 7 bands), merge orders, DER_SNR |
| `scripts/make_smh_sessions.py`, `scripts/stitch_sessions.py` | build SMHR sessions (blue+red per exposure), normalise and stitch in SMHR |
| `scripts/rv_table_we.py` | barycentric corrections at mid-exposure with barycorrpy |
| `scripts/coadd_smhr.py`, `scripts/coadd_qa_sheets.py` | inverse-variance co-add of stitched spectra; per-exposure check sheets |
| `scripts/per_exposure_lines.py`, `scripts/caK_mg_campaign.py` | Ca II H&K / Mg I 4167 sheets for every exposure |
| `scripts/hr_diagram.py` | Gaia parallax zero-point, Bayesian distance (EDSD / Galactic / halo priors), HR diagram |
| `scripts/rv_smhr_figure.py`, `scripts/rv_curve.py` | RV-vs-time figures |
| `notebooks/ws1` … `ws4` | worksheet notebooks (outputs cleared, except `ws4/part2_stellar_parameters`) |
| `figures/` | HR diagram and distance posterior for HE 1002-0755 (Gaia DR3 only), from `scripts/hr_diagram.py` |

Environments: SMHR scripts run in the course `smhr-py3` environment (Python 3.8); everything else needs
numpy, scipy, matplotlib, astropy, plus astroquery, barycorrpy and gaiadr3-zeropoint where imported.
