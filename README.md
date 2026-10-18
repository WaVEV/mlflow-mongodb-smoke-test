# MLflow MongoDB smoke test

Smoke por HTTP y MongoDB para probar un checkout local de [mlflow-mongodb](https://github.com/WaVEV/mlflow-mongodb) contra distintas versiones de MLflow 3.5+.

El runner crea un entorno temporal por versión, instala el plugin indicado y levanta servidores y bases MongoDB aisladas por escenario. Necesita Python 3.11 o 3.12, MongoDB 8.0+ y acceso a PyPI.

```bash
export MONGODB_URI='mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000'
python3.12 smoke_test/run.py --plugin-dir ../mlflow-mongodb --versions 3.5.1

# Todas las versiones estables desde 3.5
python3.12 smoke_test/run.py --plugin-dir ../mlflow-mongodb --versions-all
```

`--plugin-dir` apunta al checkout que querés probar; no cambia su rama ni sus archivos fuente. El runner instala sus dependencias en entornos temporales. Los resultados quedan en `.cache/mlflow-smoke/`, fuera de Git.

Consultá las [instrucciones completas](smoke_test/README.md), el [inventario de cobertura](smoke_test/COVERAGE.md) y el [plan](smoke_test/PLAN.md).
