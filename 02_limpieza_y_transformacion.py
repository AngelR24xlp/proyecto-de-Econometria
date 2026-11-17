"""
02 · Limpieza y transformación.

  - Lee data/raw/series_crudas.csv (salida del módulo 01) y documenta valores faltantes.
  - Series core con faltantes -> error (no se imputan). Series no core -> interpolación lineal
    de hasta MAX_INTERP meses consecutivos, dejando constancia.
  - Deflacta series nominales (IPC Lima o índices de precios de comercio exterior).
  - Logaritmos, ajuste estacional (STL robusto sobre el log), diferencias, tasas de crecimiento.
  - Variables dummy de episodios (definidas en config.EPISODIOS; son supuestos del analista).
  - Detecta saltos atípicos (posibles cambios metodológicos / de año base) para revisión manual.
  - Exporta base consolidada CSV + XLSX (con diccionario y registro de faltantes).
"""
import numpy as np, pandas as pd
from statsmodels.tsa.seasonal import STL
from config import *


def stl_sa(ln: pd.Series) -> pd.Series:
    s = ln.dropna()
    res = STL(s, period=12, robust=True).fit()
    return (ln - res.seasonal.reindex(ln.index))


def main():
    print("== 02 · Limpieza y transformación ==")
    raw = pd.read_csv(RAW_WIDE, index_col=0, parse_dates=True).asfreq("MS")
    log = pd.read_csv(SERIES_LOG)
    core = set(log.loc[log["core"], "alias"])
    df, faltantes, notas = raw.copy(), [], []

    for c in raw.columns:
        n = int(raw[c].isna().sum())
        if n == 0:
            continue
        if c in core:
            raise SystemExit(f"Serie core '{c}' con {n} faltantes: {list(raw.index[raw[c].isna()].strftime('%Y-%m'))}. "
                             "No se imputa; revise la fuente/código.")
        runs = raw[c].isna().astype(int).groupby(raw[c].notna().cumsum()).sum().max()
        if runs <= MAX_INTERP:
            df[c] = raw[c].interpolate(limit=MAX_INTERP, limit_area="inside")
            faltantes.append(dict(serie=c, n_faltantes=n, tratamiento=f"interpolación lineal (máx {runs} meses consecutivos)"))
        else:
            faltantes.append(dict(serie=c, n_faltantes=n, tratamiento="NO imputada; se excluye de los modelos"))
            df[c] = np.nan

    # ---- deflactación
    if "cpi" in df:
        for c in SERIES_REALES_DEFLACTAR_CPI:
            if c in df: df[f"{c}_real"] = df[c] / (df["cpi"] / df["cpi"].mean())
    for nom, deflt in DEFLACTAR_EXPORT.items():
        if nom in df and deflt in df: df[f"{nom}_real"] = df[nom] / (df[deflt] / 100)
    # las series nominales con sufijo propio ya no se usan en logs; se conservan en la base

    # ---- logs, ajuste estacional, diferencias
    for c in LOGS:
        if c in df and df[c].notna().sum() > 0:
            if (df[c] <= 0).any(): notas.append(f"{c}: valores <=0, no se aplica log"); continue
            df[f"l_{c}"] = np.log(df[c])
            if c in AJUSTE_ESTACIONAL:
                df[f"l_{c}_sa"] = stl_sa(df[f"l_{c}"])
    for c in [x for x in df.columns if x.startswith("l_")] + [c for c in ("tasa_ref", "tamn") if c in df]:
        df[f"d_{c}"] = df[c].diff() * (100 if c.startswith("l_") else 1)
    if "l_pbi" in df: df["crec_pbi_12m"] = df["l_pbi"].diff(12) * 100

    # ---- dummies de episodios (supuestos del analista)
    for k, (a, b, _) in EPISODIOS.items():
        df[f"D_{k.split('_')[0]}"] = ((df.index >= a) & (df.index <= b)).astype(int)
    pa, pb = PULSO_COVID
    df["P_covid"] = ((df.index >= pa) & (df.index <= pb)).astype(int)
    df["D_conflictos"] = df["D_conflictos"]            # alias explícito usado en config

    # ---- detección de saltos atípicos (para revisión de cambios metodológicos / año base / revisiones)
    atip = []
    for c in [x for x in df.columns if x.startswith("d_l_")]:
        z = (df[c] - df[c].mean()) / df[c].std()
        for f in df.index[np.abs(z) > 4]:
            atip.append(dict(variable=c, fecha=f.strftime("%Y-%m"), var_pct=round(df.loc[f, c], 2), z=round(z[f], 1),
                             comentario="REVISAR: ¿shock real (p.ej. COVID) o cambio metodológico/año base?"))
    pd.DataFrame(atip).to_csv(DATA_PROC / "saltos_atipicos_a_revisar.csv", index=False, encoding="utf-8-sig")

    df = df.loc[FECHA_INI:FECHA_FIN]
    df.rename_axis("fecha").to_csv(BASE_CSV, encoding="utf-8-sig")

    dic = log[["alias", "codigo", "descripcion", "nombre_oficial", "estado", "n_obs", "primera", "ultima", "fecha_descarga"]].copy()
    derivadas = [dict(alias=c, descripcion="variable derivada (deflactada / log / desestacionalizada STL / diferencia / dummy)") for c in df.columns if c not in raw.columns]
    with pd.ExcelWriter(BASE_XLSX, engine="openpyxl") as xw:
        df.rename_axis("fecha").to_excel(xw, sheet_name="base_mensual")
        dic.to_excel(xw, sheet_name="diccionario_series", index=False)
        pd.DataFrame(derivadas).to_excel(xw, sheet_name="variables_derivadas", index=False)
        pd.DataFrame(faltantes or [dict(serie="(ninguna)", n_faltantes=0, tratamiento="-")]).to_excel(xw, sheet_name="faltantes", index=False)
        pd.DataFrame(atip or [dict(variable="(ninguna)")]).to_excel(xw, sheet_name="saltos_a_revisar", index=False)
        pd.DataFrame({"nota": notas or ["(sin notas)"]}).to_excel(xw, sheet_name="notas", index=False)
    print(f"  Base final: {df.shape[0]} meses x {df.shape[1]} variables -> {BASE_CSV.name} / {BASE_XLSX.name}")
    print(f"  Saltos atípicos a revisar: {len(atip)}  (data/processed/saltos_atipicos_a_revisar.csv)")


if __name__ == "__main__":
    main()
