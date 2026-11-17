"""04 · Raíces unitarias: ADF, Phillips-Perron, KPSS (+ Zivot-Andrews con ruptura endógena) en niveles y primeras diferencias."""
import warnings, numpy as np, pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron, ZivotAndrews
from config import *
from econ_utils import *

warnings.filterwarnings("ignore")
TODAS = sorted(set([Y] + X_CORE + ["l_cobre", "l_imp_bk_real_sa", "l_cemento_sa", "l_credito_real"] + VAR_VARS))


def pruebas(x: pd.Series, trend: str):
    x = x.dropna(); out = {}
    a = adfuller(x, regression=trend, autolag="AIC"); out.update(ADF_stat=a[0], ADF_p=a[1], ADF_rezagos=a[2])
    p = PhillipsPerron(x, trend=trend); out.update(PP_stat=p.stat, PP_p=p.pvalue)
    k = kpss(x, regression=trend, nlags="auto"); out.update(KPSS_stat=k[0], KPSS_p=k[1])
    try:
        z = ZivotAndrews(x, trend="ct" if trend == "ct" else "c"); out.update(ZA_stat=z.stat, ZA_p=z.pvalue)
    except Exception:
        out.update(ZA_stat=np.nan, ZA_p=np.nan)
    return out


def estacionaria(r, a=0.05):
    return (r["ADF_p"] < a) and (r["PP_p"] < a) and (r["KPSS_p"] > a)


def main():
    print("== 04 · Pruebas de estacionariedad ==")
    df = cargar_base(); filas = []; orden = {}
    for v in [v for v in TODAS if v in df]:
        trend_n = "c" if v in ("tasa_ref", "tamn") else "ct"
        niv = pruebas(df[v], trend_n); filas.append(dict(variable=v, transformacion="nivel", deterministico=trend_n, **niv))
        dv = df[v].diff()
        dif = pruebas(dv, "c"); filas.append(dict(variable=v, transformacion="1ª diferencia", deterministico="c", **dif))
        if estacionaria(niv): orden[v] = "I(0)"
        elif estacionaria(dif): orden[v] = "I(1)"
        elif (niv["ADF_p"] < .05) != (niv["KPSS_p"] > .05) or (niv["PP_p"] < .05) != (niv["KPSS_p"] > .05):
            orden[v] = "ambigua (ADF/PP vs KPSS discrepan en nivel); tratar como I(1) con cautela" if estacionaria(dif) else "revisar"
        else: orden[v] = "revisar (posible I(2) o quiebre)"
    t = pd.DataFrame(filas).round(4)
    guardar_tabla(t, "estacionariedad_detalle", index=False)
    o = pd.Series(orden, name="orden_integracion").to_frame()
    guardar_tabla(o, "estacionariedad_resumen")
    guardar_resultado("orden_integracion", orden)
    print(o.to_string())
    print("  Regla: I(0) si ADF y PP rechazan raíz unitaria (5%) y KPSS no rechaza estacionariedad; I(1) si lo cumple la 1ª diferencia.")


if __name__ == "__main__":
    main()
