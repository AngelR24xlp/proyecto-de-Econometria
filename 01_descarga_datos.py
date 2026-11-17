"""
01 · Descarga de datos oficiales (BCRPData API).

Qué hace
  1. Obtiene (o lee) el catálogo de metadatos oficial del BCRP.
  2. Resuelve el código de cada serie de config.SERIES y lo CONTRASTA con su nombre oficial.
     Estados: OK_VERIFICADO, OK_OVERRIDE, OK_COINCIDE_REGEX, AUTO_UNICO, REQUIERE_REVISION, SIN_METADATA.
  3. Descarga cada serie mensual vía API JSON y guarda el JSON crudo (trazabilidad / revisiones).
  4. Consolida data/raw/series_crudas.csv y data/raw/series_log.csv (nombre, código, estado, nº obs, faltantes).

Ninguna observación se imputa ni se inventa aquí. Si una serie core no puede resolverse, el script
termina con código 2 y explica cómo fijar el código en catalogo/series_override.csv.
"""
import io, re, sys, json, time, datetime as dt
import numpy as np, pandas as pd, requests
from config import *

HEAD = {"User-Agent": "Mozilla/5.0 (investigacion-academica; pbi-peru-econometria)"}
MESES = {"ene": 1, "jan": 1, "feb": 2, "mar": 3, "abr": 4, "apr": 4, "may": 5, "jun": 6, "jul": 7,
         "ago": 8, "aug": 8, "set": 9, "sep": 9, "oct": 10, "nov": 11, "dic": 12, "dec": 12}


def _api_fecha(s):                      # "2010-01" -> "2010-1"  (formato exigido por la API)
    y, m = s.split("-"); return f"{y}-{int(m)}"


def parse_periodo(txt):
    m = re.match(r"^([A-Za-z]{3})\.?\s*(\d{2,4})$", str(txt).strip())
    if not m or m.group(1).lower() not in MESES:
        raise ValueError(f"Periodo no reconocido: {txt!r}")
    y = int(m.group(2)); y = y + 2000 if y < 100 else y
    return pd.Timestamp(y, MESES[m.group(1).lower()], 1)


# --------------------------------------------------------------------------- metadatos
def cargar_metadatos():
    txt = None
    if METADATA_PATH.exists():
        raw = METADATA_PATH.read_bytes()
    else:
        raw = None
        for u in METADATA_URLS:
            try:
                r = requests.get(u, headers=HEAD, timeout=60); r.raise_for_status()
                if len(r.content) > 5000:
                    raw = r.content; METADATA_PATH.write_bytes(raw); break
            except Exception as e:
                print(f"  (metadatos) no se pudo descargar {u}: {e}")
    if raw is None:
        print("  ! Sin metadatos. Descargue el CSV desde https://estadisticas.bcrp.gob.pe/estadisticas/series/ayuda/metadatos\n"
              f"    y guárdelo como {METADATA_PATH}")
        return None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            txt = raw.decode(enc); break
        except UnicodeDecodeError:
            continue
    meta = pd.read_csv(io.StringIO(txt), sep=None, engine="python")
    low = {c: c.lower() for c in meta.columns}
    pick = lambda *keys: next((c for c, l in low.items() if any(k in l for k in keys)), None)
    cols = dict(codigo=pick("código", "codigo"), nombre=pick("nombre"), grupo=pick("grupo"),
                frec=pick("frecuencia"), cat=pick("categor"), upd=pick("actualiz"))
    out = pd.DataFrame({k: (meta[v].astype(str) if v else "") for k, v in cols.items()})
    out["codigo"] = out["codigo"].str.strip()
    return out


# --------------------------------------------------------------------------- resolución de códigos
def resolver(serie, meta, override):
    alias, code, rx = serie["alias"], override.get(serie["alias"], serie["codigo"]), re.compile(serie["regex"], re.I)
    res = dict(alias=alias, descripcion=serie["desc"], core=serie["core"], codigo=code, nombre_oficial="",
               estado="", alternativas="")
    if meta is None:
        res["estado"] = "SIN_METADATA" if code else "REQUIERE_REVISION"
        return res
    mensual = meta["frec"].str.contains("mensual", case=False, na=False) | (meta["frec"] == "")
    if code:
        fila = meta[meta["codigo"] == code]
        if len(fila):
            nom = fila.iloc[0]["nombre"]; res["nombre_oficial"] = f"{fila.iloc[0]['grupo']} | {nom}"
            if alias in override:                 res["estado"] = "OK_OVERRIDE"; return res
            if serie["verificado"]:               res["estado"] = "OK_VERIFICADO"; return res
            if rx.search(nom + " " + fila.iloc[0]["grupo"]): res["estado"] = "OK_COINCIDE_REGEX"; return res
            res["estado"] = "REQUIERE_REVISION"
        else:
            res["estado"] = "REQUIERE_REVISION"; res["nombre_oficial"] = "(código no figura en metadatos)"
    hits = meta[mensual & (meta["nombre"] + " " + meta["grupo"]).str.contains(rx, na=False)]
    if not code and len(hits) == 1:
        h = hits.iloc[0]; res.update(codigo=h["codigo"], nombre_oficial=f"{h['grupo']} | {h['nombre']}", estado="AUTO_UNICO")
        return res
    if not res["estado"]: res["estado"] = "REQUIERE_REVISION"
    res["alternativas"] = " || ".join(f"{h.codigo}: {h.grupo} | {h.nombre}" for h in hits.head(15).itertuples())
    return res


# --------------------------------------------------------------------------- descarga
def descargar(code):
    url = f"{API_URL}/{code}/json/{_api_fecha(FECHA_INI)}/{_api_fecha(FECHA_FIN)}/esp"
    r = requests.get(url, headers=HEAD, timeout=90); r.raise_for_status()
    j = json.loads(r.content.decode("utf-8-sig"))
    per = j.get("periods") or j.get("period")
    if per is None:
        raise ValueError(f"Respuesta inesperada de la API para {code}: claves {list(j)}")
    idx, val = [], []
    for p in per:
        idx.append(parse_periodo(p["name"]))
        v = str(p["values"][0]).strip().replace(",", "")
        val.append(np.nan if v in ("n.d.", "", "nd", "None") else float(v))
    return pd.Series(val, index=pd.DatetimeIndex(idx), name=code), j, url


def main():
    print("== 01 · Descarga BCRPData ==")
    meta = cargar_metadatos()
    override = {}
    if OVERRIDE_PATH.exists():
        o = pd.read_csv(OVERRIDE_PATH, dtype=str).dropna()
        override = dict(zip(o["alias"], o["codigo"]))
    (DATA_RAW / "json").mkdir(exist_ok=True)

    log, cols = [], {}
    for s in SERIES:
        r = resolver(s, meta, override)
        r.update(n_obs=0, primera=None, ultima=None, n_faltantes=None, fecha_descarga=dt.datetime.now().isoformat(timespec="seconds"), error="")
        if r["estado"].startswith("OK") or r["estado"] in ("AUTO_UNICO", "SIN_METADATA"):
            try:
                ser, j, url = descargar(r["codigo"]); time.sleep(0.4)
                (DATA_RAW / "json" / f"{s['alias']}_{r['codigo']}.json").write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
                full = ser.reindex(pd.date_range(FECHA_INI + "-01", FECHA_FIN + "-01", freq="MS"))
                cols[s["alias"]] = full
                r.update(n_obs=int(ser.notna().sum()), primera=ser.dropna().index.min(), ultima=ser.dropna().index.max(),
                         n_faltantes=int(full.isna().sum()), url=url)
            except Exception as e:
                r["error"] = f"{type(e).__name__}: {e}"; r["estado"] += "|ERROR_DESCARGA"
        print(f"  {s['alias']:<14} {str(r['codigo']):<11} {r['estado']:<22} obs={r['n_obs']}")
        log.append(r)

    lg = pd.DataFrame(log); lg.to_csv(SERIES_LOG, index=False, encoding="utf-8-sig")
    if cols:
        pd.DataFrame(cols).rename_axis("fecha").to_csv(RAW_WIDE, encoding="utf-8-sig")
    pend = lg[(lg["alternativas"] != "") | lg["estado"].str.contains("REQUIERE|ERROR")]
    if len(pend):
        pend[["alias", "codigo", "estado", "nombre_oficial", "alternativas", "error"]].to_csv(DATA_RAW / "series_pendientes.csv", index=False, encoding="utf-8-sig")
        print(f"\n  {len(pend)} serie(s) requieren revisión -> {DATA_RAW/'series_pendientes.csv'}\n"
              f"  Fije el código correcto en {OVERRIDE_PATH} (columnas: alias,codigo) y vuelva a ejecutar.")
    faltan_core = lg[lg["core"] & ~lg["alias"].isin(cols)]
    if len(faltan_core):
        print("\n  ERROR: faltan series CORE:", ", ".join(faltan_core["alias"])); sys.exit(2)
    print("\n  Listo. Revise data/raw/series_log.csv (nombre oficial de cada código usado).")


if __name__ == "__main__":
    main()
