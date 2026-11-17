"""
07 · Impulso-respuesta y descomposición de varianza.

Usa el sistema y el rango de cointegración determinados en el módulo 05:
  - rango 0  -> VAR en primeras diferencias; IRF ortogonalizadas (Cholesky en el orden de config.VAR_VARS),
                acumuladas para leer la respuesta en NIVEL del PBI; bandas por Monte Carlo (5%-95%).
  - rango >0 -> VECM; IRF del VECM. La FEVD se calcula sobre el VAR en niveles equivalente (aproximación, se indica).
El orden de Cholesky es un SUPUESTO de identificación (de lo más exógeno a lo más endógeno); documente y pruebe órdenes alternativos.
"""
import warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
from config import *
from econ_utils import *

warnings.filterwarnings("ignore")
H = IRF_HORIZONTE


def main():
    print("== 07 · Impulso-respuesta y FEVD ==")
    df = cargar_base(); S = leer_resultado("sistema")
    if not S:
        print("  Sin sistema VAR/VECM (faltan variables). Nada que hacer."); return
    dums = [c for c in DUMMIES_PRINCIPAL if c in df]; vars_ = VAR_VARS; i_y = vars_.index(Y)
    p, rank = S["p"], S["rank"]
    if rank == 0:
        cols = [f"d_{v}" for v in vars_]; Dd = df[cols].dropna()
        res = VAR(Dd, exog=df.loc[Dd.index, dums] if dums else None).fit(S["lags"])
        irf = res.irf(H); fev = res.fevd(H)
        acumula = True
        irfs = irf.orth_irfs                                   # (H+1, k, k) respuesta i a choque j
        cum = np.cumsum(irfs, axis=0)
        try:
            lo, hi = irf.cum_errband_mc(orth=True, repl=500, signif=0.10, seed=SEED)
        except Exception as ex:
            print("  Bandas no disponibles:", ex); lo = hi = None
        etiqueta = "respuesta acumulada (nivel) del PBI s.a., puntos porcentuales"
        base_irf = cum
    else:
        Dv = df[vars_].dropna()
        m = VECM(Dv, k_ar_diff=S["lags"], coint_rank=rank, deterministic="ci", exog=df.loc[Dv.index, dums] if dums else None).fit()
        irf = m.irf(H); irfs = irf.orth_irfs; base_irf = irfs; lo = hi = None; acumula = False
        var_niv = VAR(Dv).fit(p); fev = var_niv.fevd(H)
        etiqueta = "respuesta del PBI s.a. (nivel, log×1; ver nota VECM)"
    nombres = vars_

    # --- IRF del PBI a cada choque
    rows = []
    for j, sh in enumerate(nombres):
        for h in range(H + 1):
            rows.append(dict(choque=sh, horizonte=h, irf=base_irf[h, i_y, j] * (1 if acumula else 1),
                             inf=lo[h, i_y, j] if lo is not None else np.nan, sup=hi[h, i_y, j] if hi is not None else np.nan))
    tab = pd.DataFrame(rows); guardar_tabla(tab, "irf_pbi", index=False)
    kk = len(nombres) - 1; fig, axs = plt.subplots(2, int(np.ceil(kk / 2)), figsize=(11, 5.5)); axs = axs.ravel()
    for ax, sh in zip(axs, [v for v in nombres if v != Y]):
        t = tab[tab["choque"] == sh]; ax.plot(t["horizonte"], t["irf"], lw=1.6)
        if t["inf"].notna().any(): ax.fill_between(t["horizonte"], t["inf"], t["sup"], alpha=.2)
        ax.axhline(0, c="k", lw=.6); ax.set_title(f"Choque en {sh}", fontsize=8); ax.set_xlabel("meses")
    for ax in axs[kk:]: ax.axis("off")
    fig.suptitle(f"PBI: {etiqueta} (choque de 1 desv. est.; Cholesky; banda 90% MC)", fontsize=9); guardar_fig(fig, "fig08_irf_pbi")

    # --- FEVD del PBI
    dec = fev.decomp[i_y]                                      # (H, k) proporciones
    f = pd.DataFrame(dec, columns=nombres, index=range(1, H + 1)); f.index.name = "horizonte"
    guardar_tabla(f.loc[[1, 3, 6, 12, 18, 24]].round(4), "fevd_pbi")
    fig, ax = plt.subplots(figsize=(8, 3.6)); f.plot.area(ax=ax, alpha=.85, lw=0); ax.set_ylim(0, 1)
    ax.set_title("Descomposición de varianza del error de pronóstico del PBI s.a."); ax.legend(fontsize=7, ncol=3); guardar_fig(fig, "fig09_fevd_pbi")
    guardar_resultado("irf_fevd", dict(tipo=("VAR en diferencias" if acumula else "VECM (FEVD sobre VAR en niveles, aprox.)"),
                                       fevd_h12=f.loc[12].to_dict(), fevd_h24=f.loc[24].to_dict(), orden_cholesky=nombres))
    print("  FEVD (h=12):", {k: round(v, 3) for k, v in f.loc[12].items()})


if __name__ == "__main__":
    main()
