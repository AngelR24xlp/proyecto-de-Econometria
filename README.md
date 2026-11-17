# Determinantes macroeconómicos del PBI del Perú, 2010-2025 

Autor: Ángel Ricardo Marchan Collazos (GitHub: AngelR24xlp).

> Todo resultado sale de datos descargados de la API oficial BCRPData y de estimaciones ejecutadas por este código.

## Estado de verificación (léase antes de ejecutar)

| Elemento | Estado |
|---|---|
| Formato de la API BCRPData (`.../series/api/[códigos]/json/[ini]/[fin]/[idioma]`, máx. 10 códigos por consulta) | Verificado contra la documentación del BCRP |
| Código de **Términos de intercambio** `PN38923BM` (índice 2007=100, mensual, desde ene-1996) y de los índices de precios de exportaciones `PN38915BM` e importaciones `PN38919BM` | Verificados contra el listado público de BCRPData |
| Resto de códigos de `config.py` | **CANDIDATOS.** El módulo 01 los contrasta con el nombre oficial de los metadatos del BCRP y registra el resultado en `data/raw/series_log.csv`. Si el nombre no coincide, la serie queda en `REQUIERE_REVISION` y hay que fijar el código en `catalogo/series_override.csv` |
| Pipeline 02-09 | Probado de punta a punta con datos **sintéticos** (`tests/smoke_test.py`). Esa prueba solo demuestra que el código corre; sus números no significan nada |

## Instalación y ejecución

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py                                       # 01 -> 09
```

1. El módulo 01 descarga los metadatos del BCRP. Si falla la descarga automática, baje el CSV desde
   https://estadisticas.bcrp.gob.pe/estadisticas/series/ayuda/metadatos y guárdelo como `data/raw/metadatos_bcrpdata.csv`.
2. Revise `data/raw/series_log.csv` (nombre oficial de cada código) y `data/raw/series_pendientes.csv`
   (alternativas encontradas para las series sin código). Complete `catalogo/series_override.csv` (`alias,codigo`) y reanude con
   `python run_all.py --desde 01`.
3. Revise `data/processed/saltos_atipicos_a_revisar.csv`: posibles cambios metodológicos, de año base o revisiones.
   **Verifique en BCRPData/INEI si el año base del PBI (2007=100) cambió dentro de 2010-2025** antes de interpretar resultados.

## Módulos

| Módulo | Función |
|---|---|
| `01_descarga_datos.py` | Metadatos, resolución y validación de códigos, descarga JSON, bitácora (fecha de descarga, nº de obs., faltantes) |
| `02_limpieza_y_transformacion.py` | Faltantes (core: error; no core: interpolación ≤3 meses documentada), deflactación, logs, ajuste estacional STL, dummies de episodios, detección de saltos, base CSV/XLSX |
| `03_analisis_exploratorio.py` | Descriptivos, correlaciones, VIF, gráficos 01-05 |
| `04_pruebas_estacionariedad.py` | ADF, Phillips-Perron, KPSS y Zivot-Andrews; orden de integración |
| `05_estimacion_modelos.py` | **ARDL/UECM + bounds test (principal)**; robustez (AIC/HQ, sin dummies, cobre, sector real, DOLS); rezagos distribuidos; Johansen, VAR/VECM, Granger |
| `06_diagnostico_econometrico.py` | Autocorrelación, heterocedasticidad, normalidad, RESET, CUSUM, sup-F (bootstrap), rupturas secuenciales, Chow, interacciones por episodio |
| `07_impulso_respuesta.py` | IRF (Cholesky) con bandas Monte Carlo y descomposición de varianza |
| `08_graficos_y_resultados.py` | Gráficos 10-14 a 300 dpi |
| `09_generacion_tablas.py` | `outputs/resultados_estimaciones.xlsx` y `outputs/resultados/tokens_paper.json` |

## Decisiones metodológicas que debe conocer

- **Modelo principal:** ARDL en niveles (logs) con reparametrización UECM. Admite regresores I(0) e I(1) (no I(2)) y separa corto y largo plazo.
  Rezagos por BIC. `UECM` exige orden ≥ 1 en cada regresor; cuando el criterio elige 0, se fuerza a 1 y queda registrado en `ardl_seleccion_rezagos.csv`.
- **Bounds test:** caso 3 (constante libre, sin tendencia). Las dummies entran como regresores fijos; los valores críticos asintóticos no las contemplan.
- **Desestacionalización:** STL robusto sobre el log, para PBI y series con estacionalidad marcada. Un choque tan grande como el COVID puede contaminar la componente estacional; contraste con X-13 si es relevante.
- **Rupturas:** `Bai-Perron` estricto no está en statsmodels. Se implementa un procedimiento secuencial basado en sup-F con p-valor bootstrap (aproximación). Para Bai-Perron exacto use `strucchange` en R.
- **Episodios:** las fechas de `config.EPISODIOS` son supuestos del analista y se pueden editar. Las interacciones por episodio son muchas pruebas; se reportan junto con el total evaluado para controlar la sobreinterpretación.
- **VECM:** si el rango de cointegración es > 0, la FEVD se calcula sobre el VAR en niveles equivalente (aproximación indicada en el resultado).
- **Identificación de IRF:** orden de Cholesky de `config.VAR_VARS` (supuesto). Pruebe órdenes alternativos.
- **Variables sensibles a endogeneidad:** cemento, producción minera y manufacturera forman parte del PBI; se usan solo en la especificación ampliada y con cautela.

## Prueba de humo

```bash
python tests/smoke_test.py
```
Genera datos sintéticos en `smoke_run/` y ejecuta los módulos 02-09. **No citar ni usar esos resultados.**

