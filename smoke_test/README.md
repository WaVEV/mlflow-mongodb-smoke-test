# Smoke de MLflow por HTTP y MongoDB

El runner instala el plugin del checkout indicado con `--plugin-dir` en un entorno temporal por versión de MLflow.
Levanta su propio servidor en un puerto libre y utiliza una base MongoDB nueva por escenario.
No usa el servidor de `localhost:5000`, ni borra la base indicada en tu URI.

Necesitás MongoDB 8.0+ accesible y Python 3.10+ con `venv` y acceso a los paquetes de PyPI
o tu mirror. Se recomienda Python 3.11 o 3.12 para recorrer releases antiguos y actuales con el
mismo intérprete. Docker es opcional: un MongoDB que ya tengas levantado sirve.

Cloná este repositorio y el plugin por separado. Desde la raíz de este repositorio,
indicá la ruta al checkout del plugin que querés probar. Puede contener cambios sin commit.
Arrancá por una versión:

```bash
export MONGODB_URI='mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000'
python3.12 smoke_test/run.py --plugin-dir ../mlflow-mongodb --versions 3.5.1
```

Después, toda la matriz de versiones estables 3.x desde 3.5, incluidos sus parches:

```bash
python3.12 smoke_test/run.py --plugin-dir ../mlflow-mongodb --versions-all
```

No hace falta activar la `.venv` del proyecto ni instalar pytest en el entorno del runner.
El runner utiliza la biblioteca estándar e instala MLflow, el plugin y sus herramientas
en entornos temporales. Los entornos se eliminan al terminar cada versión. Los escenarios
usan pytest, pero están fuera de `tests/` y no entran en la suite habitual del plugin.
Cliente y servidor tienen la misma versión de MLflow; esto no prueba clientes de distinta versión.

## Recortar una ejecución

```bash
# Ver versiones sin instalar ni tocar MongoDB.
python3.12 smoke_test/run.py --plugin-dir ../mlflow-mongodb --versions-all --list

# Experimentos solamente, incluidos los casos obligatorios de tags.
python3.12 smoke_test/run.py --plugin-dir ../mlflow-mongodb --versions 3.5.1 --family experiments

# Traces y assessments en dos versiones.
python3.12 smoke_test/run.py --plugin-dir ../mlflow-mongodb --versions 3.5.1 3.17.1 --family traces assessments

# Un caso por su nombre, con su propio servidor y base nueva.
python3.12 smoke_test/run.py --plugin-dir ../mlflow-mongodb --versions 3.5.1 --case simple_attributes_artifacts_and_restart

# Conservar únicamente bases de escenarios fallidos para inspeccionarlas.
python3.12 smoke_test/run.py --plugin-dir ../mlflow-mongodb --versions 3.5.1 --family experiments --keep-failed-db
```

Familias: `experiments`, `runs`, `traces`, `assessments`, `logged_models`, `registry`,
`prompts`, `links`, `metrics`, `diagnostics`. Sin `--family`, se seleccionan todas.
`--case` es una expresión `pytest -k`, no un ID del plan. Las variantes con estructura
compartida están parametrizadas; cada variante recibe servidor y base propios.
La matriz completa levanta muchos servidores y puede tardar bastante.

`--python` permite elegir el intérprete para los entornos. `--case-timeout` limita cada
escenario (240 segundos por defecto), `--startup-timeout` el arranque (90 segundos),
`--version-timeout` cada ejecución de pytest (7200 segundos) y `--install-timeout`
cada comando de preparación (900 segundos). El manejo de procesos y timeouts está
orientado a macOS/Linux. Tras el flush de exportación, los escenarios de traces esperan
hasta 30 segundos por el payload completo y sus spans Mongo; otros errores públicos se propagan.

## Qué comprueba

- Creación por cliente y REST, lifecycle, errores, tags y búsquedas de experimentos.
- Runs: logging sync/async, inmutabilidad de params, valores no finitos, historial y resumen
  vigente, inputs de datasets/modelos, outputs, estado, borrado/restauración y búsquedas.
- Traces: spans/decoradores, jerarquía, atributos, errores, metadata, logging tardío,
  reentrega, tags, búsquedas y borrados con cascada a spans/assessments.
- Recuperación completa desde artifacts locales en 3.5 o desde tracking store cuando
  corresponda; reinicio del servidor y lectura desde un proceso cliente nuevo.
- Assessments: valores nativos, updates, metadata, errores y recuperación, overrides y borrado.
- Logged models: metadata, tags, métricas de distintos runs/datasets, filtros sobre historial,
  orden por métricas, paginación y borrado lógico.
- Registry: modelos/versiones, descripciones, tags, rename, stages, latest versions, aliases,
  redacción al borrar versiones, copias con artifacts locales y asignación concurrente de versiones.
- Prompts: metadata/versiones, tags, aliases, render, búsquedas, borrado y vínculos.
- Métricas agregadas de traces/spans/assessments cuando upstream expone la API: filtros,
  dimensiones, buckets UTC, percentiles, tokens y costos disponibles.

Todas las familias de búsqueda implementadas tienen casos. El objetivo no es recorrer
cada combinación posible de campos/comparadores ni sustituir la suite funcional.
Los resultados se comparan con datos conocidos del escenario; MongoDB se consulta para
verificar persistencia, sin usar pipelines del plugin como resultado esperado.
El transporte `RestStore` usado en algunos casos es el cliente HTTP de MLflow; nunca se
instancian los stores MongoDB del plugin desde los tests.

Los ejemplos viejos se adaptaron a escenarios autónomos, sin ejecutar los archivos
originales con su URI fija:

| Script anterior | Escenario equivalente |
| --- | --- |
| `trace_test.py`, `test.py` | `simple_attributes_artifacts_and_restart`, `decorated_nested_script_equivalent` |
| `trace_test2.py` | `decorated_nested_script_equivalent` |
| `test3.py` | `simple_attributes_artifacts_and_restart` |
| `test4.py` | `error_script_equivalent` |
| `test5.py` | `experiment_isolation_script_equivalent` |
| `mlflow_smoke_dataset_inputs.py` | `dataset_inputs_and_fluent_script_equivalent` |

## Reportes y fallos conocidos

El runner muestra progreso por caso. Guarda todo en `.cache/mlflow-smoke/run-*/`:

- `summary.json`: matriz, filtros elegidos, huella del código, estado, duración y conteos.
- `<version>/client.log`, `install.log`, `packages.txt`, `junit.xml`, `cases.json`.
- `<version>/capabilities.json`: firmas y rutas REST de la instalación upstream.
- Un directorio por escenario con `scenario.json`, `server.log`, `evidence.jsonl` y artifacts.
  `scenario.json` permite relacionar el directorio con el caso y su base de datos;
  `server.json` registra el PID y la URL del servidor de ese escenario.

`evidence.jsonl` registra método/ruta/respuesta HTTP y consultas/resultados Mongo.
Cada consulta Mongo incluye un campo `mongosh` que podés pegar en la CLI: usa la base
generada y Extended JSON para conservar tipos como `ObjectId`. Para inspeccionarla después
de un fallo, ejecutá con `--keep-failed-db`; normalmente esa base ya se habrá limpiado.
No registra headers de autenticación. La URI Mongo se pasa al servidor por entorno;
los logs del servidor reemplazan esa URI por un marcador. Usá datos de smoke, no datos reales.
La huella incluye los archivos Python del checkout del plugin indicado y del smoke, más setup/pyproject, incluso
si no están commiteados. Si cambian durante una versión, se informa `SOURCE_CHANGED`.

Los estados por caso distinguen `PASSED`, `FAILED`, `SETUP_FAILED`, `UNAVAILABLE_UPSTREAM`
y `NOT_IMPLEMENTED`. El reporte conserva la fase y el error de cada fallo. Un escenario
fallido no detiene las otras variantes o familias. Los pasos posteriores dentro del mismo
test no se ejecutan y no se contabilizan como pruebas aprobadas. Los skips de funciones
no implementadas son inventario visible, no cobertura exitosa.

Los tres casos de crear/modificar/borrar tags de experimentos son obligatorios: no tienen
skip ni xfail. Los tres pasaron en la primera ejecución con MLflow 3.5.1. Se consulta Mongo
también cuando la respuesta HTTP demuestra un no-op. El smoke no modifica el store para
hacer pasar los escenarios.

Las comprobaciones de persistencia inspeccionan campos de los documentos y spans, sin
serializar los `ObjectId` con `json.dumps`. Los valores numéricos de assessments se comparan
sin exigir identidad `int`/`float`: protobuf puede convertirlos a `double` antes del HTTP;
los booleanos, strings, listas y diccionarios conservan comprobaciones de tipo, también anidadas.
Los fixtures de búsqueda de registry crean runs reales y usan sources remotos de metadata,
sin descargar ni subir artifacts a S3.

Las búsquedas de logged models especifican nombre y digest de dataset. En 3.5.1, el handler
HTTP convierte un digest omitido en `None`, que luego se usa como filtro; estos escenarios no
acreditan búsquedas por nombre solamente. Los límites inválidos de logged models usan `-1`,
porque upstream normaliza `0` al valor por defecto. Los de traces usan `RestStore` para que
la solicitud llegue por HTTP, sin el retorno vacío anticipado de `MlflowClient` con límite `0`.
El rechazo de rename sobre un experimento borrado comprueba el JSON HTTP directamente:
`INVALID_STATE` corresponde a HTTP 500 y el adaptador de reintentos del SDK oculta ese cuerpo.

También hay dos comprobaciones obligatorias del esquema de tags de registry: exigen
`{key,value}` para registered models y model versions, según la excepción acordada para
model registry (incluidos prompts). Las entradas del tracking store siguen usando `{k,v}`.
El runner no cambia esquemas ni migra datos. Los casos funcionales de tags verifican
contenido por separado; no reemplazan las comprobaciones del esquema.

La búsqueda de prompts usa filtros y paginación sin asumir que `search_prompts` acepta
`order_by`; el conjunto de nombres se compara independientemente del orden. Un alias de
prompt ausente debe devolver `INVALID_PARAMETER_VALUE`. Los filtros de timestamps de model
versions se ejecutan únicamente si el validador upstream admite ambos campos; en 3.5.1
figuran como `UNAVAILABLE_UPSTREAM`, antes de crear el servidor y la base del escenario.

Las APIs ausentes en una versión upstream se omiten con una razón concreta. Los filtros
avanzados de traces y `model_id`, finalización de logged models y unlink de traces figuran
como limitaciones del plugin. No se transforma una excepción inesperada en un skip.
`diagnostics` comprueba el contrato vacío de `search_datasets`; eso no acredita soporte
de almacenamiento de evaluation datasets.

El código de salida es 0 si todos los casos seleccionados ejecutados pasan; 1 ante fallos
o problemas de entorno, y 130 si se interrumpe. Revisá también los conteos de APIs ausentes
y funciones no implementadas: un 0 no significa paridad completa con MLflow.

La limpieza normal y por Ctrl+C apaga el servidor y elimina solo la base generada por ese
escenario. `--keep-failed-db` conserva la base fallida y registra su nombre, manteniendo el
servidor apagado. Un cierre forzado del sistema puede impedir el teardown: los nombres
exactos de las bases y directorios quedan en los reportes para su inspección.

## Límites

La primera ejecución real con MLflow 3.5.1 produjo 133 casos aprobados, 38 fallos y 20 fallos
de setup, además de 44 APIs ausentes y 4 funciones no implementadas. Se corrigieron los
fixtures y las aserciones del smoke identificados en esa ejecución y se ejecutaron nuevamente:
la segunda ejecución produjo 167 casos aprobados, 24 fallos y
ningún fallo de setup, con los mismos 44 casos upstream ausentes y 4 no implementados.
Tras corregir el smoke y los conflictos de placeholders, listas protobuf y updates de
assessments del plugin, una ejecución completa de 3.5.1 aprobó 190 casos sin fallos:
45 quedaron como `UNAVAILABLE_UPSTREAM` y 4 como `NOT_IMPLEMENTED`. Esto no acredita otras
versiones ni las funciones omitidas. Leer/parsear la fuente no verifica el contrato HTTP en cada MLflow.
El [plan](PLAN.md) mantiene el inventario más
amplio y la [cobertura](COVERAGE.md) distingue escenarios implementados de extensiones.
No implementa fixes del plugin, proveedor propio de artifacts, S3/proxy, workspaces,
webhooks, archivado de traces, fallos de drivers, límites BSON ni pruebas de carga.
