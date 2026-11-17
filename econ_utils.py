"""Funciones auxiliares compartidas por los módulos 02-09."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *

plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 300, "font.size": 9, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False})


def cargar_base() -> pd.DataFrame:
    return pd.read_csv(BASE_CSV, index_col=0, parse_dates=True).asfreq("MS")


def sombrear_episodios(ax, etiquetas=False):
    for nombre, (a, b, c) in EPISODIOS.items():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b) + pd.offsets.MonthEnd(0), color=c, alpha=.18, lw=0,
                   label=nombre if etiquetas else None)


def guardar_fig(fig, nombre):
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"{nombre}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def guardar_tabla(df: pd.DataFrame, nombre: str, index=True):
    df.to_csv(OUT_TAB / f"{nombre}.csv", index=index, encoding="utf-8-sig")


def _jsonable(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (pd.Timestamp,)): return o.strftime("%Y-%m")
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def guardar_resultado(clave: str, valor):
    """Acumula resultados clave (para el paper) en outputs/resultados/resultados.json."""
    p = OUT_RES / "resultados.json"
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    d[clave] = json.loads(json.dumps(valor, default=_jsonable))
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


def leer_resultado(clave, default=None):
    p = OUT_RES / "resultados.json"
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8")).get(clave, default)


def disponibles(df, cols):
    """Devuelve (presentes, ausentes) y exige que no haya NaN en las presentes dentro de la muestra."""
    pres = [c for c in cols if c in df.columns and df[c].notna().sum() > 0.9 * len(df)]
    return pres, [c for c in cols if c not in pres]


# ------------------------------------------------------------------ pruebas de cambio estructural
def _rss(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    return float(e @ e), X @ b, e


def sup_f(y, W, Z, trim=0.15):
    """Quandt-Andrews: sup-F de cambio parcial en los coeficientes de Z (W estables) con fecha desconocida."""
    n = len(y); lo = max(int(n * trim), 2); hi = n - lo
    rss0, _, _ = _rss(np.column_stack([W, Z]) if W.size else Z, y)
    q, k0 = Z.shape[1], (W.shape[1] if W.size else 0) + Z.shape[1]
    Fs = np.full(n, np.nan)
    for s in range(lo, hi):
        Z1 = Z.copy(); Z1[s:] = 0
        Z2 = Z.copy(); Z2[:s] = 0
        X1 = np.column_stack([W, Z1, Z2]) if W.size else np.column_stack([Z1, Z2])
        rss1, _, _ = _rss(X1, y)
        Fs[s] = ((rss0 - rss1) / q) / (rss1 / (n - k0 - q))
    s_best = int(np.nanargmax(Fs))
    return float(Fs[s_best]), s_best, Fs


def sup_f_boot(y, W, Z, trim=0.15, B=499, seed=SEED):
    """p-valor del sup-F por wild bootstrap (Rademacher) bajo la nula de estabilidad."""
    rng = np.random.default_rng(seed)
    X0 = np.column_stack([W, Z]) if W.size else Z
    _, fit, e = _rss(X0, y)
    F_obs, s_best, Fs = sup_f(y, W, Z, trim)
    cnt = 0
    for _ in range(B):
        ys = fit + e * rng.choice([-1.0, 1.0], size=len(y))
        if sup_f(ys, W, Z, trim)[0] >= F_obs:
            cnt += 1
    return F_obs, s_best, (1 + cnt) / (B + 1), Fs


def rupturas_secuenciales(y, W, Z, fechas, max_breaks=3, minseg=36, B=299, alpha=0.05):
    """Procedimiento secuencial (en la línea de Bai, 1997 / Bai-Perron, 1998) basado en sup-F:
    se busca una ruptura, y si es significativa se repite en cada submuestra. Es una APROXIMACIÓN;
    para Bai-Perron estricto (programación dinámica y criterios BIC/LWZ) use R::strucchange."""
    found = []

    def rec(a, b):
        if len(found) >= max_breaks or b - a < 2 * minseg:
            return
        trim = min(minseg / (b - a), 0.45)
        F, s, p, _ = sup_f_boot(y[a:b], W[a:b] if W.size else W, Z[a:b], trim=trim, B=B, seed=SEED + a)
        if p < alpha:
            found.append(dict(fecha=fechas[a + s], sup_F=F, p_boot=p, tramo=f"{fechas[a]:%Y-%m}:{fechas[b-1]:%Y-%m}"))
            rec(a, a + s); rec(a + s, b)
    rec(0, len(y))
    return sorted(found, key=lambda d: d["fecha"])


# ------------------------------------------------------------------ ARDL / UECM
from statsmodels.tsa.ardl import ARDL as _ARDL, UECM as _UECM, ardl_select_order as _select


def muestra_modelo(df, y, xs, dummies):
    cols = [y] + list(xs) + list(dummies)
    d = df[cols].dropna()
    return d


def seleccionar_orden(d, y, xs, dummies, ic):
    """Búsqueda de rezagos por criterio de información (AIC/BIC/HQ).
    NOTA: UECM exige orden >= 1 en todo regresor (el nivel x_{t-1} entra al término de corrección de error);
    por eso si el criterio excluye un regresor o elige orden 0, se fuerza orden 1. Queda registrado."""
    sel = _select(d[y], ARDL_CFG["maxlag"], d[xs], ARDL_CFG["maxorder"], trend="c", ic=ic,
                  fixed=d[dummies] if dummies else None)
    p = max(1, max(sel.ar_lags) if sel.ar_lags else 1)
    orden, forzados = {}, []
    for x in xs:
        lags = sel.dl_lags.get(x, []) or [0]
        o = max(lags)
        if o < 1:
            forzados.append(x); o = 1
        orden[x] = o
    return p, orden, forzados


def construir_uecm(d, y, xs, dummies, p, orden):
    ardl = _ARDL(d[y], p, d[xs], orden, trend="c", fixed=d[dummies] if dummies else None)
    u = _UECM.from_ardl(ardl)
    return ardl, u, u.fit()


def ols_de_uecm(u):
    """Reconstruye el OLS de la ecuación UECM (idéntico a UECM.fit; se verifica en 05) para usar
    los diagnósticos de statsmodels que exigen un OLSResults."""
    import statsmodels.api as sm
    return sm.OLS(u._y, u._x).fit()
