"""
06 · Diagnóstico econométrico, estabilidad estructural y análisis de episodios.

  1. Residuos del ECM principal: autocorrelación (Breusch-Godfrey, Ljung-Box, Durbin-Watson), heterocedasticidad
     (Breusch-Pagan/Koenker, ARCH-LM), normalidad (Jarque-Bera), forma funcional (RESET).
  2. Estabilidad: CUSUM (OLS y recursivo), Quandt-Andrews sup-F con p-valor bootstrap, rupturas secuenciales
     (aproximación a Bai-Perron), Chow en fechas de inicio/fin de cada episodio.
  3. Episodios: (a) residuo medio por episodio; (b) ¿cambió la relación? interacciones episodio x término de largo plazo
     y episodio x impacto contemporáneo, con errores robustos HC1.
Las fechas de los episodios son supuestos del analista (config.EPISODIOS), no hechos estadísticos.
"""
import warnings, numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import (acorr_breusch_godfrey, acorr_ljungbox, het_breuschpagan, het_arch,
                                          linear_reset, breaks_cusumolsresid, recursive_olsresiduals)
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from config import *
from econ_utils import *
from econ_utils import _rss

warnings.filterwarnings("ignore")
ALFA = 0.05


def fila(prueba, h0, est, p):
    return dict(prueba=prueba, H0=h0, estadistico=est, p_valor=p, decision_5pct="No rechaza H0" if p > ALFA else "Rechaza H0")


def main():
    print("== 06 · Diagnóstico y estabilidad ==")
    df = cargar_base(); R = leer_resultado("modelo_principal")
    d = muestra_modelo(df, R["y"], R["x"], R["dummies"])
    ardl, u, r = construir_uecm(d, R["y"], R["x"], R["dummies"], R["p"], R["orden"])
    ols = ols_de_uecm(u); e = pd.Series(ols.resid, index=r.resid.index); k = ols.model.exog.shape[1]
    names = list(r.params.index); X = np.asarray(u._x); yv = np.asarray(u._y); fechas = e.index

    # ------------------------------------------------ 1. residuos
    T = []
    for L in (1, 6, 12):
        lm, p, f, fp = acorr_breusch_godfrey(ols, nlags=L); T.append(fila(f"Breusch-Godfrey LM (rezagos={L})", "Sin autocorrelación hasta L", lm, p))
    lb = acorr_ljungbox(e, lags=[12], return_df=True); T.append(fila("Ljung-Box Q (12)", "Residuos ruido blanco", lb["lb_stat"].iloc[0], lb["lb_pvalue"].iloc[0]))
    T.append(dict(prueba="Durbin-Watson", H0="(referencia ≈2)", estadistico=durbin_watson(e), p_valor=np.nan, decision_5pct="-"))
    lm, p, f, fp = het_breuschpagan(e, ols.model.exog); T.append(fila("Breusch-Pagan/Koenker", "Homocedasticidad", lm, p))
    lm, p, f, fp = het_arch(e, nlags=12); T.append(fila("ARCH-LM (12)", "Sin efectos ARCH", lm, p))
    jb, p, sk, ku = jarque_bera(e); T.append(fila(f"Jarque-Bera (asim={sk:.2f}, curt={ku:.2f})", "Normalidad", jb, p))
    rs = linear_reset(ols, power=2, test_type="fitted", use_f=True); T.append(fila("RESET de Ramsey (potencia 2)", "Forma funcional correcta", float(rs.fvalue), float(rs.pvalue)))
    cs, pcs, _ = breaks_cusumolsresid(np.asarray(e), ddof=k); T.append(fila("CUSUM (Ploberger-Krämer, OLS)", "Parámetros estables", cs, pcs))
    tabla = pd.DataFrame(T); guardar_tabla(tabla, "diagnostico_residuos", index=False)
    print(tabla[["prueba", "estadistico", "p_valor"]].round(4).to_string(index=False))

    # CUSUM recursivo (gráfico)
    try:
        # las dummies de choque son 0 al inicio de la muestra (matriz inicial singular): el CUSUM recursivo se calcula sin ellas
        cols_rec = [i for i, n_ in enumerate(names) if n_ not in R["dummies"]]
        ols_rec = sm.OLS(yv, X[:, cols_rec]).fit()
        rr = recursive_olsresiduals(ols_rec, skip=len(cols_rec) + 10, alpha=1 - ALFA)
        ci = np.asarray(rr[6]); cus = np.asarray(rr[5])[-ci.shape[1]:]; fx = fechas[-len(cus):]
        fig, ax = plt.subplots(figsize=(8, 3.2)); ax.plot(fx, cus, lw=1.3, label="CUSUM"); ax.plot(fx, ci[0], "r--", lw=.8); ax.plot(fx, ci[1], "r--", lw=.8, label="banda 5%")
        sombrear_episodios(ax); ax.set_title("CUSUM recursivo del ECM principal"); ax.legend(); guardar_fig(fig, "fig06_cusum")
        cus_fuera = bool(np.any((cus > ci[1]) | (cus < ci[0])))
    except Exception as ex:
        cus_fuera = None; print("  CUSUM recursivo no disponible:", ex)

    # ------------------------------------------------ 2. estabilidad
    niv = [i for i, n in enumerate(names) if n == "const" or (n.endswith(".L1") and not n.startswith("D."))]
    est = [i for i in range(len(names)) if i not in niv]
    Z, W = X[:, niv], X[:, est]
    F, s, pb, Fs = sup_f_boot(yv, W, Z, trim=0.15, B=499)
    pd.DataFrame({"fecha": fechas, "F": Fs}).to_csv(OUT_RES / "supF_trayectoria.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 3)); ax.plot(fechas, Fs, lw=1.2); ax.axvline(fechas[s], c="r", ls="--", lw=.8)
    sombrear_episodios(ax); ax.set_title(f"Estadístico F de Chow para cada fecha candidata (sup-F={F:.1f}, p bootstrap={pb:.3f})")
    guardar_fig(fig, "fig07_supF")
    guardar_tabla(pd.DataFrame([dict(sup_F=F, fecha_ruptura=fechas[s].strftime("%Y-%m"), p_bootstrap=pb, B=499, recorte=0.15,
                                     bloque_inestable="constante + términos de nivel (ECT y largo plazo)")]), "estabilidad_supF", index=False)
    seq = rupturas_secuenciales(yv, W, Z, fechas, max_breaks=3, minseg=36, B=299)
    guardar_tabla(pd.DataFrame(seq or [dict(fecha="(ninguna significativa al 5%)")]), "estabilidad_rupturas_secuenciales", index=False)

    n = len(yv); q = Z.shape[1]; k0 = X.shape[1]; chow = []
    for nombre, (a, b, _) in EPISODIOS.items():
        for lab, f0 in (("inicio", pd.Timestamp(a)), ("fin+1", pd.Timestamp(b) + pd.offsets.MonthBegin(1))):
            if f0 <= fechas[10] or f0 >= fechas[-10]: continue
            s0 = int(np.searchsorted(fechas, f0))
            Z1, Z2 = Z.copy(), Z.copy(); Z1[s0:] = 0; Z2[:s0] = 0
            r1, *_ = _rss(np.column_stack([W, Z1, Z2]), yv); r0, *_ = _rss(X[:, est + niv], yv)
            Fc = ((r0 - r1) / q) / (r1 / (n - k0 - q))
            chow.append(dict(episodio=nombre, fecha_corte=f"{f0:%Y-%m}", tipo=lab, F_Chow=Fc, gl=f"({q},{n-k0-q})", p_F=1 - stats.f.cdf(Fc, q, n - k0 - q)))
    guardar_tabla(pd.DataFrame(chow), "estabilidad_chow_episodios", index=False)

    # ------------------------------------------------ 3. episodios
    dumcols = {"desaceleracion_2014_16": "D_desaceleracion", "crisis_politica_2017_19": "D_crisis", "covid_2020": "D_covid",
               "recuperacion_2021": "D_recuperacion", "conflictos_2022_23": "D_conflictos", "evolucion_2024_25": "D_evolucion"}
    res_ep, inter = [], []
    sd = e.std()
    for nombre, col in dumcols.items():
        m = df.loc[fechas, col].values.astype(bool)
        if m.sum() < 3: continue
        mod = sm.OLS(e.values[m], np.ones(m.sum())).fit(cov_type="HAC", cov_kwds=dict(maxlags=min(3, m.sum() - 1)))
        res_ep.append(dict(episodio=nombre, n_meses=int(m.sum()), residuo_medio=float(mod.params[0]), residuo_medio_en_sd=float(mod.params[0] / sd), p_HAC=float(mod.pvalues[0])))
        Dm = m.astype(float)
        objetivos = [(n_, "largo plazo / ajuste (nivel t-1)") for n_ in names if n_.endswith(".L1") and not n_.startswith("D.")] + \
                    [(n_, "impacto contemporáneo (Δ t)") for n_ in names if n_.startswith("D.") and n_.endswith(".L0")]
        for n_, tipo in objetivos:
            j = names.index(n_); Xi = np.column_stack([X, Dm * X[:, j]])
            if np.linalg.matrix_rank(Xi) < Xi.shape[1]: continue
            f_ = sm.OLS(yv, Xi).fit(cov_type="HC1")
            inter.append(dict(episodio=nombre, termino=n_, tipo=tipo, coef_base=float(f_.params[j]), coef_interaccion=float(f_.params[-1]), p_HC1=float(f_.pvalues[-1])))
    guardar_tabla(pd.DataFrame(res_ep), "episodios_residuo_medio", index=False)
    I = pd.DataFrame(inter)
    if len(I):
        I["significativa_5pct"] = I["p_HC1"] < ALFA; guardar_tabla(I, "episodios_interacciones", index=False)
    guardar_resultado("diagnostico", dict(
        supF=F, supF_fecha=fechas[s], supF_p=pb, rupturas_secuenciales=seq, cusum_p=pcs, cusum_recursivo_fuera_de_banda=cus_fuera,
        n_interacciones_significativas=int(I["significativa_5pct"].sum()) if len(I) else 0, n_interacciones_evaluadas=len(I)))
    print("  Estabilidad y episodios: tablas guardadas.")
    if len(I):
        print(f"  Interacciones significativas al 5%: {int(I['significativa_5pct'].sum())} de {len(I)} "
              "(con tantas pruebas se esperan falsos positivos; interprete con cautela).")


if __name__ == "__main__":
    main()
