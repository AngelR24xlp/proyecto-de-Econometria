"""
05 · Estimación.

  A. MODELO PRINCIPAL: ARDL en niveles (logs) -> UECM/ECM con contraste de cointegración por límites
     (Pesaran, Shin & Smith, 2001). Rezagos por BIC (AIC y HQ como robustez). Dummies de choque como regresores fijos.
  B. ROBUSTEZ del largo plazo: otros criterios de rezago, sin dummies, cobre en vez de términos de intercambio,
     especificación ampliada (sector real) y DOLS (Stock & Watson, 1993).
  C. REZAGOS DISTRIBUIDOS en variaciones mensuales (efectos retardados y multiplicadores acumulados, errores HAC).
  D. SISTEMA: selección de rezagos VAR, Johansen (trace / max-eig), VAR en diferencias o VECM según el rango,
     causalidad de Granger. (IRF y FEVD en el módulo 07.)
Todo resultado se calcula aquí a partir de la base; nada se escribe a mano.
"""
import warnings, numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from config import *
from econ_utils import *

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ A/B. ARDL-UECM
def estimar_spec(df, nombre, sp):
    xs, _ = disponibles(df, sp["x"]); dums = [c for c in sp["dummies"] if c in df]
    if len(xs) < len(sp["x"]):
        print(f"  [{nombre}] omitida: faltan {set(sp['x']) - set(xs)}"); return None
    d = muestra_modelo(df, sp["y"], xs, dums)
    p, orden, forz = seleccionar_orden(d, sp["y"], xs, dums, sp["ic"])
    ardl, u, r = construir_uecm(d, sp["y"], xs, dums, p, orden)
    ra = ardl.fit()
    bt = r.bounds_test(case=ARDL_CFG["case"])
    a = r.params[f"{sp['y']}.L1"]
    lr = pd.DataFrame({"coef": r.ci_params, "se": r.ci_bse, "t": r.ci_tvalues, "p": r.ci_pvalues})
    lr[["ic95_inf", "ic95_sup"]] = r.ci_conf_int().values
    lr = lr.drop(index=[sp["y"]], errors="ignore")
    return dict(nombre=nombre, d=d, xs=xs, dums=dums, p=p, orden=orden, forzados=forz, ardl=ardl, u=u, r=r, bt=bt,
                alpha=a, alpha_p=r.pvalues[f"{sp['y']}.L1"], lr=lr, aic=ra.aic, bic=ra.bic, hqic=ra.hqic)


def dols(d, y, xs, dums, k=2, hac=6):
    X = d[xs].copy()
    for x in xs:
        dx = d[x].diff()
        for j in range(-k, k + 1):
            X[f"d_{x}_{j}"] = dx.shift(-j)
    Z = pd.concat([d[y], X, d[dums] if dums else None], axis=1).dropna()
    m = sm.OLS(Z[y], sm.add_constant(Z.drop(columns=y))).fit(cov_type="HAC", cov_kwds=dict(maxlags=hac))
    return pd.DataFrame({"coef": m.params[xs], "se": m.bse[xs], "p": m.pvalues[xs]}), m.nobs


# ------------------------------------------------------------------ C. rezagos distribuidos
def rezagos_distribuidos(df, xs, dums):
    dy = f"d_{Y}"; dX = [f"d_{x}" for x in xs]
    base = df[[dy] + dX + dums].copy()
    for L in range(DL_MAXLAGS + 1):
        pass
    def diseno(L, idx=None):
        cols = {f"{v}_L{j}": base[v].shift(j) for v in dX for j in range(L + 1)}
        X = pd.DataFrame(cols); X[dums] = base[dums]
        return X
    comun = diseno(DL_MAXLAGS).dropna().index.intersection(base[dy].dropna().index)
    ics = []
    for L in range(DL_MAXLAGS + 1):
        X = sm.add_constant(diseno(L).loc[comun]); m = sm.OLS(base.loc[comun, dy], X).fit()
        ics.append(dict(L=L, k=int(m.df_model) + 1, AIC=m.aic, BIC=m.bic, HQ=m.info_criteria("hqic")))
    ics = pd.DataFrame(ics).set_index("L")
    L = int(ics["BIC"].idxmin())
    X = sm.add_constant(diseno(L).dropna()); yy = base.loc[X.index, dy]
    m = sm.OLS(yy, X).fit(cov_type="HAC", cov_kwds=dict(maxlags=6))
    coefs = pd.DataFrame({"coef": m.params, "se": m.bse, "p": m.pvalues}).drop(index="const")
    filas = []
    for v in dX:
        nombres = [f"{v}_L{j}" for j in range(L + 1)]
        R = np.zeros((1, len(m.params))); 
        for nm in nombres: R[0, list(m.params.index).index(nm)] = 1
        w = m.wald_test(R, use_f=True, scalar=True)
        filas.append(dict(variable=v, multiplicador_acumulado=float(m.params[nombres].sum()),
                          se=float(np.sqrt(R @ m.cov_params().values @ R.T)[0, 0]), p_wald=float(w.pvalue)))
    return ics, L, coefs, pd.DataFrame(filas), m


# ------------------------------------------------------------------ D. VAR / VECM
def sistema(df):
    vv, falt = disponibles(df, VAR_VARS)
    if falt:
        print("  Sistema VAR omitido; faltan:", falt); return None
    Dv = df[VAR_VARS].dropna()
    so = VAR(Dv).select_order(VAR_MAXLAGS)
    ics = pd.DataFrame(so.ics).rename_axis("rezagos")
    sel = {k: int(v) for k, v in so.selected_orders.items()}
    p = max(VAR_MIN_LAGS, sel["bic"])                       # p = rezagos del VAR en niveles
    def johansen(D, k_ar):
        j = coint_johansen(D, det_order=0, k_ar_diff=k_ar)
        tab = pd.DataFrame({"H0: rango <=": range(len(j.lr1)), "traza": j.lr1, "cv_traza_5%": j.cvt[:, 1],
                            "max_eig": j.lr2, "cv_maxeig_5%": j.cvm[:, 1]})
        r_tr = 0
        for i in range(len(j.lr1)):
            if j.lr1[i] > j.cvt[i, 1]: r_tr += 1
            else: break
        r_me = 0
        for i in range(len(j.lr2)):
            if j.lr2[i] > j.cvm[i, 1]: r_me += 1
            else: break
        return tab, r_tr, r_me
    tab, r_tr, r_me = johansen(Dv, p - 1)
    tab_pre, r_tr_pre, r_me_pre = johansen(Dv.loc[:"2019-12-01"], p - 1)
    rank = r_tr
    out = dict(p=p, rank=rank, rank_trace=r_tr, rank_maxeig=r_me, rank_trace_pre2020=r_tr_pre, rank_maxeig_pre2020=r_me_pre,
               seleccion_ic=sel, tabla_johansen=tab, tabla_johansen_pre=tab_pre, ics=ics, vars=VAR_VARS)
    dums = [c for c in DUMMIES_PRINCIPAL if c in df]
    if rank == 0:
        Dd = df[[f"d_{v}" for v in VAR_VARS]].dropna()
        lags = max(1, p - 1)
        res = VAR(Dd, exog=df.loc[Dd.index, dums] if dums else None).fit(lags)
        out.update(tipo="VAR en primeras diferencias", lags=lags, estable=bool(res.is_stable()))
        pbi = f"d_{Y}"; g = []
        for v in [c for c in Dd.columns if c != pbi]:
            t1 = res.test_causality(pbi, [v], kind="f"); t2 = res.test_causality(v, [pbi], kind="f")
            g.append(dict(causa=v, efecto=pbi, F=t1.test_statistic, p=t1.pvalue))
            g.append(dict(causa=pbi, efecto=v, F=t2.test_statistic, p=t2.pvalue))
        out["granger"] = pd.DataFrame(g)
    else:
        m = VECM(Dv, k_ar_diff=p - 1, coint_rank=rank, deterministic="ci",
                 exog=df.loc[Dv.index, dums] if dums else None).fit()
        b = pd.DataFrame(m.beta, index=VAR_VARS, columns=[f"beta{i+1}" for i in range(rank)])
        a = pd.DataFrame(m.alpha, index=VAR_VARS, columns=[f"alpha{i+1}" for i in range(rank)])
        a_p = pd.DataFrame(m.pvalues_alpha, index=VAR_VARS, columns=[f"p_alpha{i+1}" for i in range(rank)])
        i_y = VAR_VARS.index(Y)
        lp = (-b.iloc[:, 0] / b.iloc[i_y, 0]).rename("elasticidad_LP_sobre_PBI (1er vector, normalizado en PBI)")
        out.update(tipo="VECM", lags=p - 1, beta=b, alpha=a, p_alpha=a_p, largo_plazo=lp.drop(index=Y).to_frame())
        # Granger sobre el VAR en niveles equivalente (aprox.)
        res = VAR(Dv).fit(p); g = []
        for v in [c for c in VAR_VARS if c != Y]:
            t1 = res.test_causality(Y, [v], kind="f"); t2 = res.test_causality(v, [Y], kind="f")
            g.append(dict(causa=v, efecto=Y, F=t1.test_statistic, p=t1.pvalue))
            g.append(dict(causa=Y, efecto=v, F=t2.test_statistic, p=t2.pvalue))
        out["granger"] = pd.DataFrame(g)
    return out


def main():
    print("== 05 · Estimación de modelos ==")
    df = cargar_base()
    res_specs, bounds, lr_all, orden_rows, alpha_rows = {}, [], [], [], []
    for nombre, sp in ESPECIFICACIONES.items():
        e = estimar_spec(df, nombre, sp)
        if e is None: continue
        res_specs[nombre] = e
        orden_rows.append(dict(spec=nombre, criterio=sp["ic"], p_y=e["p"], **{f"q_{k}": v for k, v in e["orden"].items()},
                               orden_forzado_a_1=",".join(e["forzados"]) or "-", n_obs=int(e["r"].nobs), AIC=e["aic"], BIC=e["bic"], HQ=e["hqic"]))
        cv = e["bt"].critical_values
        bounds.append(dict(spec=nombre, F_estadistico=e["bt"].statistic, p_I0=e["bt"].pvalue["lower"], p_I1=e["bt"].pvalue["upper"],
                           cv5_I0=cv.loc[95.0, "lower"], cv5_I1=cv.loc[95.0, "upper"], cv1_I0=cv.loc[99.0, "lower"], cv1_I1=cv.loc[99.0, "upper"],
                           caso=f"Caso {ARDL_CFG['case']} (constante libre, sin tendencia)"))
        hl = np.log(.5) / np.log(1 + e["alpha"]) if -1 < e["alpha"] < 0 else np.nan
        alpha_rows.append(dict(spec=nombre, alpha_ECT=e["alpha"], p=e["alpha_p"], vida_media_meses=hl))
        t = e["lr"].copy(); t.insert(0, "spec", nombre); t.index.name = "variable"; lr_all.append(t.reset_index())
        print(f"  [{nombre}] p={e['p']} orden={e['orden']} alpha={e['alpha']:.3f}  Bounds F={e['bt'].statistic:.2f}")

    if "principal" not in res_specs:
        raise SystemExit("No se pudo estimar el modelo principal (faltan series core).")
    P = res_specs["principal"]

    guardar_tabla(pd.DataFrame(orden_rows), "ardl_seleccion_rezagos", index=False)
    guardar_tabla(pd.DataFrame(bounds), "ardl_bounds_test", index=False)
    guardar_tabla(pd.DataFrame(alpha_rows), "ecm_velocidad_ajuste", index=False)
    guardar_tabla(pd.concat(lr_all), "ardl_largo_plazo_todas_specs", index=False)
    r = P["r"]
    cf = pd.DataFrame({"coef": r.params, "se": r.bse, "t": r.tvalues, "p": r.pvalues}); cf.index.name = "parametro"
    guardar_tabla(cf, "ecm_principal_coeficientes")
    lr = P["lr"].copy(); lr.index.name = "variable"
    guardar_tabla(lr, "ardl_largo_plazo_principal")

    # series para gráficos / diagnósticos
    pd.DataFrame({"dy_obs": pd.Series(P["u"]._y, index=r.resid.index), "resid": r.resid,
                  "dy_fit": pd.Series(P["u"]._y, index=r.resid.index) - r.resid,
                  "desvio_equilibrio_LP": r.ci_resids.reindex(r.resid.index)}).to_csv(OUT_RES / "uecm_principal_series.csv")

    # chequeo interno: el OLS reconstruido replica el UECM
    chk = float(np.max(np.abs(np.asarray(ols_de_uecm(P["u"]).params) - np.asarray(r.params))))
    assert chk < 1e-8, "El OLS reconstruido no coincide con UECM.fit()"

    # DOLS
    dl, nobs = dols(P["d"], Y, P["xs"], P["dums"])
    dl.index.name = "variable"; guardar_tabla(dl, "dols_largo_plazo")
    # Rezagos distribuidos
    ics, Lopt, coefs, acum, mdl = rezagos_distribuidos(df, P["xs"], P["dums"])
    guardar_tabla(ics, "rezagos_distribuidos_seleccion"); guardar_tabla(coefs, "rezagos_distribuidos_coef")
    guardar_tabla(acum, "rezagos_distribuidos_acumulados", index=False)
    # Sistema
    S = sistema(df)
    if S:
        guardar_tabla(S["ics"], "var_seleccion_rezagos"); guardar_tabla(S["tabla_johansen"], "johansen_muestra_completa", index=False)
        guardar_tabla(S["tabla_johansen_pre"], "johansen_pre_2020", index=False)
        guardar_tabla(S["granger"], "granger", index=False)
        if S["tipo"] == "VECM":
            guardar_tabla(S["beta"], "vecm_beta"); guardar_tabla(S["alpha"], "vecm_alpha"); guardar_tabla(S["largo_plazo"], "vecm_largo_plazo")
        guardar_resultado("sistema", {k: S[k] for k in ("p", "rank", "rank_trace", "rank_maxeig", "rank_trace_pre2020", "rank_maxeig_pre2020", "tipo", "lags", "seleccion_ic")})

    guardar_resultado("modelo_principal", dict(
        y=Y, x=P["xs"], dummies=P["dums"], p=P["p"], orden=P["orden"], orden_forzado=P["forzados"], n_obs=int(r.nobs),
        muestra=f"{P['d'].index[0]:%Y-%m} a {P['d'].index[-1]:%Y-%m}", alpha=P["alpha"], alpha_p=P["alpha_p"],
        bounds_F=P["bt"].statistic, bounds_p_I1=P["bt"].pvalue["upper"], largo_plazo=P["lr"][["coef", "se", "p"]].to_dict("index"),
        dols=dl.to_dict("index"), dl_L=Lopt))
    print("  Modelo principal:", f"p={P['p']}, orden={P['orden']}, N={int(r.nobs)}")
    print("  Tablas escritas en", OUT_TAB)


if __name__ == "__main__":
    main()
