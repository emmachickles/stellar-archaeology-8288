#!/usr/bin/env python
"""Distance posterior, absolute magnitude and HR diagram for HE 1002-0755.

Run in the wd-periodicity env (needs gaiadr3-zeropoint):
    ~/miniforge3-fresh/envs/wd-periodicity/bin/python scripts/hr_diagram.py

Background: Gaia DR3, random subsample with parallax > 0.3 mas, parallax/error > 10, RUWE < 1.4, G < 15 (colours not de-reddened)
Inputs : data/gaia/he1002_gaia.ecsv, background_hr.ecsv, comparison_gaia.ecsv (Gaia DR3 via astroquery;
         comparison stars: HD 122563, CS 22892-052, HE 1523-0901, CD-38 245, G 64-12, HD 140283, IDs from SIMBAD,
         with S&F 2011 full-column E(B-V) from IRSA, scaled to each star's distance with a 125 pc dust layer)
Outputs: worksheets/ws2/part2_observation_details/plots/hr_diagram_HE1002-0755.{pdf,png}
         worksheets/ws2/part2_observation_details/plots/distance_posterior_HE1002-0755.{pdf,png}
         worksheets/ws2/part2_observation_details/results/distance_posterior.json

Likelihood : Gaussian in parallax, with the Lindegren et al. (2021) zero-point removed.
Priors     : (a) EDSD, L = 1.35 kpc;  (b) Galactic density model (Juric et al. 2008 disks + halo);
             (c) halo only (the star has [Fe/H] ~ -3).  The ordering of the three tells how much
             the prior, rather than the parallax, is deciding the answer.
Extinction : E(B-V) = 0.0446 (Schlafly & Finkbeiner 2011), A_0 = 3.1 E(B-V); Gaia EDR3 k_X(BP-RP, A_0).
"""
import os, json, warnings
import numpy as np
from astropy.table import Table
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from zero_point import zpt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PLOTS = "worksheets/ws2/part2_observation_details/plots"
RES = "worksheets/ws2/part2_observation_details/results"
EBV, EBV_ERR = 0.0446, 0.0089   # 20% on the dust column

# Gaia EDR3 extinction law: k = c1 + c2 C + c3 C^2 + c4 C^3 + c5 A0 + c6 A0^2 + c7 C A0, C = (BP-RP)_0
K = dict(G=[0.9761, -0.1704, 0.0086, 0.0011, -0.0438, 0.0013, 0.0099],
         BP=[1.1517, -0.0871, -0.0333, 0.0173, -0.0230, 0.0006, 0.0043],
         RP=[0.6104, -0.0170, -0.0026, -0.0017, -0.0078, 0.00005, 0.0006])

def kx(b, C, A0):
    c = K[b]; return c[0] + c[1]*C + c[2]*C**2 + c[3]*C**3 + c[4]*A0 + c[5]*A0**2 + c[6]*C*A0

def extinction(bprp_obs, ebv):
    A0 = 3.1*ebv; C = bprp_obs
    for _ in range(20):
        C = bprp_obs - A0*(kx("BP", C, A0) - kx("RP", C, A0))
    return A0*kx("G", C, A0), C

def gal_xyz(r, l, b):
    l, b = np.radians(l), np.radians(b)
    x = 8.2 - r*np.cos(b)*np.cos(l); y = -r*np.cos(b)*np.sin(l); z = 0.02 + r*np.sin(b)
    return x, y, z

def density(r, l, b, which):
    x, y, z = gal_xyz(r, l, b); R = np.hypot(x, y); rr = np.sqrt(R**2 + (z/0.64)**2)
    thin = np.exp(-(R-8.2)/2.6 - (abs(z)-0.02)/0.3)
    thick = 0.12*np.exp(-(R-8.2)/3.6 - (abs(z)-0.02)/0.9)
    halo = 0.0051*(8.2/np.maximum(rr, 0.1))**2.77
    return {"galactic": thin+thick+halo, "halo": halo}[which]

def summarize(r, post):
    cdf = np.cumsum(post); cdf /= cdf[-1]
    return [float(np.interp(q, cdf, r)) for q in (0.16, 0.5, 0.84)]

def zero_point(g):
    f = lambda x: np.array([x], dtype=float)           # gaiadr3-zeropoint's can_cast rejects Python scalars under numpy 2
    pc = float(g["pseudocolour"]) if np.isfinite(float(g["pseudocolour"])) else np.nan
    return float(zpt.get_zpt(f(g["phot_g_mean_mag"]), f(g["nu_eff_used_in_astrometry"]), f(pc),
                             f(g["ecl_lat"]), np.array([int(g["astrometric_params_solved"])]), _warnings=False)[0])

def ebv_to_distance(ebv_total, r_kpc, b_deg, h=0.125):
    """Fraction of the full-column (S&F 2011) reddening in front of a star at r, for a dust layer
    with scale height h (kpc). ~1 for HE 1002-0755; it matters for the nearby comparison stars."""
    return ebv_total * (1 - np.exp(-r_kpc*abs(np.sin(np.radians(b_deg)))/h))

def posterior(g, which, rmax=60.0, n=300000):
    plx, sig, zp = float(g["parallax"]), float(g["parallax_error"]), zero_point(g)
    if not np.isfinite(zp):      # Lindegren et al. 2021 is calibrated for 6 < G < 21; HD 122563 (G = 5.9) is outside
        zp = 0.0
    r = np.linspace(max(0.2/(plx - zp + 5*sig), 1e-3) if plx - zp > 5*sig else 0.005, min(rmax, 1.0/max(plx - zp - 8*sig, 1/rmax)), n)
    like = np.exp(-0.5*((plx - zp - 1.0/r)/sig)**2)
    if which == "EDSD (L=1.35 kpc)":
        pr = r**2*np.exp(-r/1.35)
    else:
        pr = r**2*density(r, float(g["l"]), float(g["b"]), "galactic" if which == "Galactic model" else "halo")
    return r, like, like*pr, zp

def MG_samples(g, r, post, ebv_total, rng, N=200000):
    G, BP, RP = (float(g[k]) for k in ("phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"))
    cdf = np.cumsum(post); cdf /= cdf[-1]
    rs = np.interp(rng.random(N), cdf, r)
    ebv = ebv_to_distance(ebv_total, np.median(rs), float(g["b"]))
    AG, C0 = extinction(BP - RP, ebv)
    MG = G - rng.normal(AG, 0.2*AG + 1e-4, rs.size) - 5*np.log10(rs*1e3) + 5
    return rs, MG, AG, C0, ebv

def main():
    rng = np.random.default_rng(1)
    zpt.load_tables()
    g = Table.read("data/gaia/he1002_gaia.ecsv")[0]
    G, BP, RP = (float(g[k]) for k in ("phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"))
    priors = ["EDSD (L=1.35 kpc)", "Galactic model", "Halo only"]
    out = dict(parallax_mas=float(g["parallax"]), parallax_err_mas=float(g["parallax_error"]), ebv=EBV, bprp_obs=BP-RP,
               bailer_jones_geo_kpc=[float(g[k])/1e3 for k in ("r_lo_geo", "r_med_geo", "r_hi_geo")],
               bailer_jones_photogeo_kpc=[float(g[k])/1e3 for k in ("r_lo_photogeo", "r_med_photogeo", "r_hi_photogeo")],
               posteriors={})
    store = {}
    for name in priors:
        r, like, post, zp = posterior(g, name)
        rs, MG, AG, C0, _ = MG_samples(g, r, post, EBV, rng)
        lo, med, hi = summarize(r, post); m16, m50, m84 = np.percentile(MG, [16, 50, 84])
        out["posteriors"][name] = dict(r_kpc=[lo, med, hi], M_G=[float(m16), float(m50), float(m84)])
        store[name] = dict(r=r, like=like, post=post, MG=MG, pct=(m16, m50, m84))
        print(f"{name:20s} r = {med:5.2f} (+{hi-med:4.2f} -{med-lo:4.2f}) kpc   M_G = {m50:+.2f} (+{m84-m50:.2f} -{m50-m16:.2f})")
    out.update(zero_point_mas=zp, A_G=AG, bprp_0=C0)
    print(f"zero-point {zp:+.4f} mas, A_G = {AG:.3f}, (BP-RP)_0 = {C0:.3f}")
    print("BJ21 geo     :", out["bailer_jones_geo_kpc"], " photogeo:", out["bailer_jones_photogeo_kpc"])

    # comparison stars: same machinery, halo prior (all are halo stars), reddening scaled to distance
    comp = Table.read("data/gaia/comparison_gaia.ecsv")
    out["comparison"] = {}
    cpts = []
    for c in comp:
        r, like, post, zpc = posterior(c, "Halo only")
        rs, MG, AGc, C0c, ebvc = MG_samples(c, r, post, float(c["ebv_sf11"]), rng)
        m16, m50, m84 = np.percentile(MG, [16, 50, 84]); rl, rm, rh = summarize(r, post)
        out["comparison"][str(c["name"])] = dict(r_kpc=[rl, rm, rh], M_G=[float(m16), float(m50), float(m84)], bprp_0=C0c,
                                                 ebv_used=ebvc, plx_over_err=float(c["parallax"]/c["parallax_error"]))
        cpts.append((str(c["name"]), C0c, m16, m50, m84))
        print(f"  {str(c['name']):13s} plx/err {float(c['parallax']/c['parallax_error']):6.1f}  r = {rm:6.3f} kpc  "
              f"E(B-V) {ebvc:.3f}  (BP-RP)_0 {C0c:.2f}  M_G = {m50:+.2f} (+{m84-m50:.2f} -{m50-m16:.2f})")
    json.dump(out, open(f"{RES}/distance_posterior.json", "w"), indent=1)

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 10, "axes.linewidth": 0.9,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "legend.frameon": False})
    col = {"EDSD (L=1.35 kpc)": "#0072B2", "Galactic model": "#D55E00", "Halo only": "#009E73"}

    # ---- HR diagram
    bg = Table.read("data/gaia/background_hr.ecsv")
    Mbg = bg["phot_g_mean_mag"] + 5*np.log10(bg["parallax"]) - 10
    fig, ax = plt.subplots(figsize=(5.4, 5.8))
    ax.hist2d(bg["bp_rp"], Mbg, bins=[200, 240], range=[[-0.3, 3.3], [-5.5, 12.5]], cmin=1,
              norm=matplotlib.colors.LogNorm(), cmap="Greys")
    off = {"EDSD (L=1.35 kpc)": -0.03, "Galactic model": 0.0, "Halo only": 0.03}
    for name in priors:
        m16, m50, m84 = store[name]["pct"]
        ax.errorbar(C0 + off[name], m50, yerr=[[m50-m16], [m84-m50]], xerr=0.015, fmt="o", color=col[name], ms=5,
                    capsize=2, zorder=5, label=f"HE 1002$-$0755, {name}")
    mk = dict(zip(["HD 122563", "CS 22892-052", "HE 1523-0901", "CD-38 245", "G 64-12", "HD 140283"], ["*", "s", "D", "^", "v", "P"]))
    for nm, c0, m16, m50, m84 in cpts:
        ax.errorbar(c0, m50, yerr=[[m50-m16], [m84-m50]], fmt=mk.get(nm, "o"), color="k", mfc="#F0E442", ms=7, capsize=2, zorder=4)
        dxy = (7, -3)
        if m50 < 0: continue
        ax.annotate(nm.replace("-", "$-$"), (c0, m50), xytext=dxy, textcoords="offset points", fontsize=7,
                    ha="right" if dxy[0] < 0 else "left")
    ax.set_xlim(-0.3, 3.3); ax.set_ylim(12.5, -5.5)
    ax.set_xlabel(r"$(G_{\rm BP}-G_{\rm RP})_0$"); ax.set_ylabel(r"$M_G$")
    # the giants pile up within 0.5 mag of each other: zoom on them
    ins = ax.inset_axes([0.57, 0.17, 0.40, 0.32])
    for name in priors:
        m16, m50, m84 = store[name]["pct"]
        ins.errorbar(C0 + off[name], m50, yerr=[[m50-m16], [m84-m50]], fmt="o", color=col[name], ms=4, capsize=2, zorder=5)
    for nm, c0, m16, m50, m84 in cpts:
        if m50 < 0:
            ins.errorbar(c0, m50, yerr=[[m50-m16], [m84-m50]], fmt=mk.get(nm, "o"), color="k", mfc="#F0E442", ms=6, capsize=2)
            ins.annotate(nm.replace("-", "$-$"), (c0, m50), xytext={"HE 1523-0901": (6, 5), "HD 122563": (6, -8)}.get(nm, (6, -2)),
                         textcoords="offset points", fontsize=6)
    ins.annotate("HE 1002$-$0755", (C0, store["Halo only"]["pct"][1]), xytext=(8, 0), textcoords="offset points", fontsize=6.5)
    ins.set_xlim(0.95, 1.85); ins.set_ylim(-0.3, -3.0); ins.tick_params(labelsize=6)
    ax.indicate_inset_zoom(ins, edgecolor="0.4")
    ax.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{PLOTS}/hr_diagram_HE1002-0755.{ext}", dpi=200)
    plt.close(fig)

    # ---- distance posterior
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    rL = store["Halo only"]["r"]; L = store["Halo only"]["like"]
    a1.plot(rL, L/np.trapezoid(L, rL), color="k", ls=":", lw=1.4, zorder=6,
            label="likelihood only (no prior)")
    for name in priors:
        rr, pp = store[name]["r"], store[name]["post"]
        a1.plot(rr, pp/np.trapezoid(pp, rr), color=col[name], lw=1.5, label=name)
        lo, med, hi = out["posteriors"][name]["r_kpc"]
        a1.axvline(med, color=col[name], lw=0.7, ls="--")
    bl, bm, bh = out["bailer_jones_geo_kpc"]
    a1.axvspan(bl, bh, color="0.85", zorder=0, label="Bailer-Jones et al. (2021) geo, 16-84%")
    a1.axvline(1/(out["parallax_mas"] - zp), color="k", lw=0.8, label=r"$1/(\varpi - \varpi_0)$")
    a1.set_xlim(0, 30); a1.set_xlabel("distance (kpc)"); a1.set_ylabel("probability density (kpc$^{-1}$)")
    a1.legend(fontsize=7)
    for name in priors:
        a2.hist(store[name]["MG"], bins=200, range=(-5, 1), density=True, histtype="step", color=col[name], lw=1.4, label=name)
    a2.set_xlim(-5, 1); a2.invert_xaxis(); a2.set_xlabel(r"$M_G$"); a2.set_ylabel("probability density (mag$^{-1}$)")
    a2.legend(fontsize=7)
    fig.suptitle(r"HE 1002$-$0755: Gaia DR3 parallax %.3f $\pm$ %.3f mas, zero-point %+.3f mas" %
                 (out["parallax_mas"], out["parallax_err_mas"], zp), fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{PLOTS}/distance_posterior_HE1002-0755.{ext}", dpi=200)

if __name__ == "__main__":
    main()
