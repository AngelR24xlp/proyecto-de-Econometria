"""
Configuración central del proyecto.
Todo lo que el investigador puede (y debe) revisar está aquí: series, códigos BCRP,
especificaciones de los modelos y definición de los episodios de choque.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SMOKE = os.environ.get("PBI_SMOKE") == "1"          # SOLO para la prueba de humo con datos sintéticos
BASE = (ROOT / "smoke_run") if SMOKE else ROOT

DATA_RAW = BASE / "data" / "raw"
DATA_PROC = BASE / "data" / "processed"
OUT = BASE / "outputs"
OUT_TAB = OUT / "tablas"
OUT_FIG = OUT / "figuras"
OUT_RES = OUT / "resultados"
CONFIG_DIR = ROOT / "catalogo"
for _d in (DATA_RAW, DATA_PROC, OUT_TAB, OUT_FIG, OUT_RES):
    _d.mkdir(parents=True, exist_ok=True)

BASE_CSV = DATA_PROC / "base_mensual_2010_2025.csv"
BASE_XLSX = DATA_PROC / "base_mensual_2010_2025.xlsx"
RAW_WIDE = DATA_RAW / "series_crudas.csv"
SERIES_LOG = DATA_RAW / "series_log.csv"
METADATA_PATH = DATA_RAW / "metadatos_bcrpdata.csv"
OVERRIDE_PATH = CONFIG_DIR / "series_override.csv"      # alias,codigo  (lo edita el usuario)

# ----------------------------------------------------------------------------- periodo
FECHA_INI = "2010-01"
FECHA_FIN = "2025-12"

# ----------------------------------------------------------------------------- API BCRPData
# Formato documentado: .../series/api/[códigos]/[formato]/[periodo inicial]/[periodo final]/[idioma]
API_URL = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"
METADATA_URLS = [   # el BCRP publica los metadatos como CSV; si la descarga automática falla,
    "https://estadisticas.bcrp.gob.pe/estadisticas/series/metadata",   # descárguelo manualmente desde
]                                                                      # .../series/ayuda/metadatos y guárdelo en METADATA_PATH

# ----------------------------------------------------------------------------- catálogo de series
# verificado=True  -> código confirmado por el autor del proyecto contra el listado público de BCRPData.
# verificado=False -> código CANDIDATO: 01_descarga_datos.py lo contrasta con el nombre en los metadatos
#                     oficiales (regex) y deja constancia en data/raw/series_log.csv. Si no coincide,
#                     la serie queda en REQUIERE_REVISION y hay que fijarla en catalogo/series_override.csv.
# core=True        -> necesaria para el modelo principal.
SERIES = [
    dict(alias="pbi",        desc="PBI real mensual (índice 2007=100)", codigo="PN01770AM", verificado=False, core=True,
         regex=r"(PBI|producto bruto interno).*(índice|indice|2007)"),
    dict(alias="tot",        desc="Términos de intercambio (índice 2007=100)", codigo="PN38923BM", verificado=True, core=True,
         regex=r"t[eé]rminos de intercambio"),
    dict(alias="px",         desc="Índice de precios nominales de exportaciones (2007=100)", codigo="PN38915BM", verificado=True, core=False,
         regex=r"exportaciones"),
    dict(alias="pm",         desc="Índice de precios nominales de importaciones (2007=100)", codigo="PN38919BM", verificado=True, core=False,
         regex=r"importaciones"),
    dict(alias="cobre",      desc="Cotización internacional del cobre (cUS$/lb)", codigo="PN01652XM", verificado=False, core=False,
         regex=r"cobre"),
    dict(alias="itcrm",      desc="Índice de tipo de cambio real multilateral", codigo="PN01258BM", verificado=False, core=True,
         regex=r"tipo de cambio real"),
    dict(alias="tasa_ref",   desc="Tasa de interés de referencia de la política monetaria (%)", codigo="PN07819NM", verificado=False, core=True,
         regex=r"referencia"),
    dict(alias="credito",    desc="Crédito al sector privado (millones S/)", codigo=None, verificado=False, core=True,
         regex=r"cr[eé]dito.*sector privado"),
    dict(alias="cpi",        desc="IPC Lima (deflactor)", codigo="PN01270PM", verificado=False, core=True,
         regex=r"(IPC|[ií]ndice de precios al consumidor)"),
    dict(alias="liquidez",   desc="Liquidez total del sector privado (millones S/)", codigo=None, verificado=False, core=False,
         regex=r"liquidez"),
    dict(alias="tamn",       desc="Tasa de interés activa en moneda nacional (TAMN, %)", codigo=None, verificado=False, core=False,
         regex=r"TAMN|tasa de inter[eé]s activa"),
    dict(alias="gasto_pub",  desc="Gasto no financiero del gobierno general (millones S/)", codigo=None, verificado=False, core=True,
         regex=r"gastos? no financieros?.*gobierno general"),
    dict(alias="inv_pub",    desc="Inversión pública / formación bruta de capital del gobierno general", codigo=None, verificado=False, core=False,
         regex=r"(inversi[oó]n|formaci[oó]n bruta de capital).*(p[uú]blic|gobierno general)"),
    dict(alias="ing_trib",   desc="Ingresos tributarios del gobierno general (millones S/)", codigo=None, verificado=False, core=False,
         regex=r"ingresos tributarios"),
    dict(alias="cemento",    desc="Consumo interno de cemento (miles TM)", codigo=None, verificado=False, core=False,
         regex=r"cemento"),
    dict(alias="exportaciones", desc="Exportaciones totales FOB (millones US$)", codigo=None, verificado=False, core=False,
         regex=r"exportaciones.*(total|fob)|total.*exportaciones"),
    dict(alias="imp_bk",     desc="Importaciones de bienes de capital (millones US$)", codigo=None, verificado=False, core=False,
         regex=r"bienes de capital"),
    dict(alias="prod_minera", desc="PBI minería metálica (índice 2007=100)", codigo=None, verificado=False, core=False,
         regex=r"miner[ií]a metálica|miner[ií]a metalica"),
    dict(alias="prod_manuf", desc="PBI manufactura (índice 2007=100)", codigo=None, verificado=False, core=False,
         regex=r"manufactura"),
    dict(alias="empleo",     desc="Empleo formal (puestos de trabajo)", codigo=None, verificado=False, core=False,
         regex=r"empleo|puestos de trabajo"),
]

# ----------------------------------------------------------------------------- transformaciones
SERIES_REALES_DEFLACTAR_CPI = ["credito", "gasto_pub", "inv_pub", "ing_trib", "liquidez"]   # nominal S/ -> real
DEFLACTAR_EXPORT = {"exportaciones": "px", "imp_bk": "pm"}                                     # nominal US$ -> real con índice de precios
LOGS = ["pbi", "tot", "cobre", "itcrm", "credito_real", "gasto_pub_real", "inv_pub_real", "ing_trib_real",
        "liquidez_real", "cemento", "exportaciones_real", "imp_bk_real", "prod_minera", "prod_manuf", "empleo"]
AJUSTE_ESTACIONAL = ["pbi", "gasto_pub_real", "inv_pub_real", "ing_trib_real", "cemento", "exportaciones_real",
                     "imp_bk_real", "prod_minera", "prod_manuf"]       # STL robusto sobre el logaritmo
MAX_INTERP = 3     # meses consecutivos máximos a interpolar en series NO core

# ----------------------------------------------------------------------------- episodios (editable)
# (inicio, fin, color) – definiciones del analista, NO son datos oficiales.
EPISODIOS = {
    "desaceleracion_2014_16": ("2014-01-01", "2016-12-01", "#bbbbbb"),
    "crisis_politica_2017_19": ("2017-01-01", "2019-12-01", "#f2c94c"),
    "covid_2020": ("2020-03-01", "2020-12-01", "#eb5757"),
    "recuperacion_2021": ("2021-01-01", "2021-12-01", "#6fcf97"),
    "conflictos_2022_23": ("2022-12-01", "2023-03-01", "#bb6bd9"),
    "evolucion_2024_25": ("2024-01-01", "2025-12-01", "#56ccf2"),
}
PULSO_COVID = ("2020-04-01", "2020-05-01")   # meses del colapso

# ----------------------------------------------------------------------------- especificaciones de modelos
Y = "l_pbi_sa"
X_CORE = ["l_tot", "l_itcrm", "tasa_ref", "l_credito_real", "l_gasto_pub_real_sa"]
DUMMIES_PRINCIPAL = ["D_covid", "P_covid", "D_conflictos"]
ARDL_CFG = dict(maxlag=4, maxorder=3, ic="bic", case=3)    # case 3 = constante sin restringir, sin tendencia

ESPECIFICACIONES = {
    "principal":            dict(y=Y, x=X_CORE, dummies=DUMMIES_PRINCIPAL, ic="bic"),
    "ic_aic":               dict(y=Y, x=X_CORE, dummies=DUMMIES_PRINCIPAL, ic="aic"),
    "ic_hq":                dict(y=Y, x=X_CORE, dummies=DUMMIES_PRINCIPAL, ic="hqic"),
    "sin_dummies":          dict(y=Y, x=X_CORE, dummies=[], ic="bic"),
    "cobre_en_lugar_de_tot": dict(y=Y, x=["l_cobre"] + X_CORE[1:], dummies=DUMMIES_PRINCIPAL, ic="bic"),
    "ampliada_sector_real": dict(y=Y, x=X_CORE + ["l_imp_bk_real_sa"], dummies=DUMMIES_PRINCIPAL, ic="bic"),
}

# Sistema VAR/VECM: orden de Cholesky de lo más exógeno a lo más endógeno
VAR_VARS = ["l_tot", "l_itcrm", "l_gasto_pub_real_sa", "l_credito_real", "tasa_ref", "l_pbi_sa"]
VAR_MAXLAGS = 12
VAR_MIN_LAGS = 2
IRF_HORIZONTE = 24
DL_MAXLAGS = 8       # rezagos distribuidos: se elige 0..DL_MAXLAGS por AIC/BIC/HQ
SEED = 12345
