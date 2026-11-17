"""Ejecuta el pipeline completo desde cero (01 -> 09).  Uso:  python run_all.py [--desde 02]"""
import subprocess, sys
from pathlib import Path
MODS = ["01_descarga_datos", "02_limpieza_y_transformacion", "03_analisis_exploratorio", "04_pruebas_estacionariedad",
        "05_estimacion_modelos", "06_diagnostico_econometrico", "07_impulso_respuesta", "08_graficos_y_resultados", "09_generacion_tablas"]
desde = sys.argv[sys.argv.index("--desde") + 1] if "--desde" in sys.argv else "01"
for m in [m for m in MODS if m[:2] >= desde]:
    print(f"\n######## {m} ########", flush=True)
    r = subprocess.run([sys.executable, str(Path(__file__).parent / f"{m}.py")])
    if r.returncode:
        sys.exit(f"Detenido en {m} (código {r.returncode}). Corrija y reanude con: python run_all.py --desde {m[:2]}")
print("\nPipeline completado. Resultados en outputs/.")
