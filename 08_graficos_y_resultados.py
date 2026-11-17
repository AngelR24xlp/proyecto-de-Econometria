"""08 · Gráficos de resultados (300 dpi): ajuste del ECM, desvío del equilibrio, robustez del largo plazo, episodios, multiplicadores."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from config import *
from econ_utils import *


def main():
    print("== 08 · Gráficos de resultados ==")
    s = pd.read_csv(OUT_RES / "uecm_principal_series.csv", index_col=0, parse_dates=True)
    fig, axs = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    axs[0].plot(s.index, s["dy_obs"], lw=1, label="Δ ln PBI s.a. (observado)"); axs[0].plot(s.index, s["dy_fit"], lw=1, label="ajustado ECM")
    axs[0].legend(fontsize=7); sombrear_episodios(axs[0]); axs[0].set_title("Ecuación de corrección de error (modelo principal)")
    axs[1].plot(s.index, s["resid"], lw=.9, c="k"); axs[1].axhline(0, c="r", lw=.6); sombrear_episodios(axs[1]); axs[1].set_title("Residuos")
    guardar_fig(fig, "fig10_ajuste_ecm")

    fig, ax = plt.subplots(figsize=(10, 3.2)); ax.plot(s.index, s["desvio_equilibrio_LP"], lw=1.2); ax.axhline(0, c="r", lw=.6)
    sombrear_episodios(ax, etiquetas=True); ax.legend(fontsize=6, ncol=3); ax.set_title("Desvío del PBI s.a. respecto de su relación de largo plazo (residuo de cointegración)")
    guardar_fig(fig, "fig11_desvio_equilibrio")

    t = pd.read_csv(OUT_TAB / "ardl_largo_plazo_todas_specs.csv"); t = t[t["variable"] != "const"]
    vars_ = list(dict.fromkeys(t["variable"])); specs = list(dict.fromkeys(t["spec"]))
    fig, axs = plt.subplots(1, len(vars_), figsize=(2.3 * len(vars_), 3.4), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, v in zip(axs, vars_):
        sub = t[t["variable"] == v].set_index("spec").reindex(specs).dropna(subset=["coef"])
        yy = np.arange(len(sub))
        ax.errorbar(sub["coef"], yy, xerr=[sub["coef"] - sub["ic95_inf"], sub["ic95_sup"] - sub["coef"]], fmt="o", ms=3, lw=1)
        ax.axvline(0, c="k", lw=.6); ax.set_title(v, fontsize=7); ax.set_yticks(yy); ax.set_yticklabels(sub.index, fontsize=6)
    fig.suptitle("Coeficientes de largo plazo (IC 95%) por especificación", fontsize=9); guardar_fig(fig, "fig12_robustez_largo_plazo")

    e = pd.read_csv(OUT_TAB / "episodios_residuo_medio.csv")
    fig, ax = plt.subplots(figsize=(7, 3)); ax.bar(e["episodio"], e["residuo_medio_en_sd"], color=["#c00" if p < .05 else "#888" for p in e["p_HAC"]])
    ax.axhline(0, c="k", lw=.6); ax.set_ylabel("residuo medio (desv. est.)"); plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    ax.set_title("Residuo medio del ECM por episodio (rojo: p HAC < 5%)"); guardar_fig(fig, "fig13_residuo_episodios")

    m = pd.read_csv(OUT_TAB / "rezagos_distribuidos_coef.csv", index_col=0)
    vs = sorted({i.rsplit("_L", 1)[0] for i in m.index if "_L" in i}); fig, axs = plt.subplots(1, len(vs), figsize=(2.6 * len(vs), 3), sharey=False)
    for ax, v in zip(np.atleast_1d(axs), vs):
        mm = m[m.index.str.startswith(v + "_L")]; lag = [int(i.rsplit("_L", 1)[1]) for i in mm.index]
        ax.errorbar(lag, mm["coef"], yerr=1.96 * mm["se"], fmt="o-", ms=3, lw=1); ax.axhline(0, c="k", lw=.6); ax.set_title(v, fontsize=7)
    fig.suptitle("Modelo de rezagos distribuidos: efecto de Δx_{t-j} sobre Δ ln PBI (IC 95% HAC)", fontsize=9); guardar_fig(fig, "fig14_rezagos_distribuidos")
    print("  Figuras 10-14 generadas en", OUT_FIG)


if __name__ == "__main__":
    main()
