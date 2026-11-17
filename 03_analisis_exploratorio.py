"""03 · Análisis exploratorio: estadística descriptiva, correlaciones, colinealidad (VIF), gráficos."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import STL
from config import *
from econ_utils import *


def main():
    print("== 03 · Análisis exploratorio ==")
    df = cargar_base()
    vars_ = [v for v in [Y] + X_CORE + ["l_cobre", "l_cemento_sa", "l_imp_bk_real_sa"] if v in df]

    d = df[vars_].describe().T
    d["asimetria"] = df[vars_].skew(); d["curtosis_exceso"] = df[vars_].kurt()
    d["JB_p"] = [stats.jarque_bera(df[v].dropna())[1] for v in vars_]
    guardar_tabla(d.round(4), "estadistica_descriptiva")

    dif = [f"d_{v}" for v in vars_ if f"d_{v}" in df]
    corr = df[dif].corr()
    guardar_tabla(corr.round(3), "correlaciones_diferencias")

    X = df[[v for v in X_CORE if v in df]].dropna()
    Xc = np.column_stack([np.ones(len(X)), X.values])
    vif = pd.Series([variance_inflation_factor(Xc, i + 1) for i in range(X.shape[1])], index=X.columns, name="VIF_niveles")
    guardar_tabla(vif.round(2).to_frame(), "vif_regresores")

    # --- gráficos
    n = len(vars_); fig, axs = plt.subplots(int(np.ceil(n / 2)), 2, figsize=(11, 2.2 * np.ceil(n / 2)))
    for ax, v in zip(axs.ravel(), vars_):
        ax.plot(df.index, df[v], lw=1.2); sombrear_episodios(ax); ax.set_title(v)
    guardar_fig(fig, "fig01_series_niveles")

    if "crec_pbi_12m" in df:
        fig, ax = plt.subplots(figsize=(10, 3.2)); ax.plot(df.index, df["crec_pbi_12m"], lw=1.4)
        sombrear_episodios(ax, etiquetas=True); ax.axhline(0, c="k", lw=.6)
        ax.set_ylabel("% anual"); ax.set_title("Crecimiento interanual del PBI mensual (calculado sobre el log del índice)")
        ax.legend(fontsize=6, ncol=3, loc="lower left"); guardar_fig(fig, "fig02_crecimiento_pbi")

    fig, ax = plt.subplots(figsize=(6, 5)); im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index, fontsize=6); ax.grid(False)
    fig.colorbar(im); ax.set_title("Correlación de variaciones mensuales"); guardar_fig(fig, "fig03_correlaciones")

    if "l_pbi" in df:
        st = STL(df["l_pbi"].dropna(), period=12, robust=True).fit(); fig = st.plot(); fig.set_size_inches(9, 6)
        guardar_fig(fig, "fig04_stl_pbi")

    if "d_l_pbi_sa" in df:
        fig, ax = plt.subplots(figsize=(10, 3.2))
        for v in [c for c in ("d_l_tot", "d_l_credito_real", "d_l_gasto_pub_real_sa") if c in df]:
            ax.plot(df.index, df["d_l_pbi_sa"].rolling(36).corr(df[v]), label=v)
        ax.axhline(0, c="k", lw=.6); sombrear_episodios(ax); ax.legend(fontsize=7)
        ax.set_title("Correlación móvil (36 m) entre variación del PBI s.a. y sus determinantes")
        guardar_fig(fig, "fig05_correlacion_movil")
    print("  Tablas y figuras 01-05 generadas.")


if __name__ == "__main__":
    main()
