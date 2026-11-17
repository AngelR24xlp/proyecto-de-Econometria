"""
09 · Consolidación de resultados.
  - outputs/resultados_estimaciones.xlsx : una hoja por tabla (CSV) + hoja 'LEEME'.
  - outputs/resultados/tokens_paper.json : valores numéricos planos que usará el generador del paper (sin escribir números a mano).
"""
import json, re, pandas as pd
from openpyxl.utils import get_column_letter
from config import *
from econ_utils import *


def aplanar(d, pref=""):
    out = {}
    for k, v in d.items():
        kk = f"{pref}{k}"
        if isinstance(v, dict): out.update(aplanar(v, kk + "."))
        else: out[kk] = v
    return out


def main():
    print("== 09 · Tablas y resultados consolidados ==")
    xlsx = OUT / "resultados_estimaciones.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        pd.DataFrame({"contenido": [
            "Resultados generados automáticamente por el pipeline (módulos 01-09).",
            f"Muestra: {FECHA_INI} a {FECHA_FIN}, frecuencia mensual. Fuente de datos: ver data/raw/series_log.csv.",
            "Cada hoja corresponde a un CSV de outputs/tablas/. Ninguna cifra fue escrita manualmente.",
            "Modo prueba de humo (PBI_SMOKE=1): datos SINTÉTICOS; no usar." if SMOKE else "Modo: datos oficiales."]}).to_excel(xw, sheet_name="LEEME", index=False)
        for f in sorted(OUT_TAB.glob("*.csv")):
            t = pd.read_csv(f); nm = re.sub(r"[^A-Za-z0-9_]", "_", f.stem)[:31]
            t.to_excel(xw, sheet_name=nm, index=False)
            ws = xw.sheets[nm]
            for i, c in enumerate(t.columns, 1):
                ws.column_dimensions[get_column_letter(i)].width = min(45, max(12, len(str(c)) + 2))
    p = OUT_RES / "resultados.json"
    tokens = aplanar(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else {}
    tokens["_modo_prueba_sintetico"] = SMOKE
    (OUT_RES / "tokens_paper.json").write_text(json.dumps(tokens, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {xlsx.name} ({len(list(OUT_TAB.glob('*.csv')))} hojas) y tokens_paper.json ({len(tokens)} valores)")


if __name__ == "__main__":
    main()
