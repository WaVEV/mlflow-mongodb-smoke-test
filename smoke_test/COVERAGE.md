# Correspondencia con el inventario

Esta tabla describe código escrito, no resultados ejecutados. Las funciones se encuentran
en `cases/test_<familia>.py`; las variantes homogéneas usan parametrización.
Los IDs del plan pueden agrupar varios comportamientos: tener un escenario para un ID
no significa haber probado todas sus combinaciones.

| Plan | Implementación |
| --- | --- |
| EXP-01–03 | Creación Python/REST, selección fluent, lectura ID/nombre y rename. |
| EXP-04–06 | Tres escenarios obligatorios e independientes de tags, sin skip ni xfail. |
| EXP-07–08 | Lifecycle/vistas y aislamiento de runs, duplicados activos/borrados, nombres inválidos, IDs ausentes y rename en conflicto. |
| RUN-01–04 | Creación Python/REST, logging individual/batch sync/async, update, tag replacement, inmutabilidad de params y error asíncrono. |
| RUN-05 | Historial paginado por HTTP, entregas fuera de orden, empate de métricas, latest por step, NaN e infinitos. |
| RUN-06–07 | Numpy fluent dataset inputs con contexto y tags anidados k/v, deduplicación, model inputs/outputs y outputs repetidos preservados. |
| RUN-08–09 | Delete/restore, búsquedas por lifecycle, conservación de historial, rechazo de escritura inactiva y batch vacío/conflictivo. |
| TRC-01–06 | Equivalentes de scripts anteriores, atributos estructurados, metadata explícita y dos experimentos aislados. |
| TRC-07–08 | Exportación asíncrona con flush, OTLP por RestStore, hijo antes de raíz, reentrega, batch completo/info donde haya transporte, trazas vacías/parciales. |
| TRC-09–11 | Tags, protección de ubicaciones frente a atributos de spans, delete por ID o timestamp/count, cascada, token usage deduplicado y metadata autoritativa. |
| ART-01–03 | Payload local en 3.5, lectura de tracking store en versiones modernas, reinicio y cliente nuevo. |
| ASM-01–05 | Feedback/Expectation nativos, igualdad numérica tras protobuf sin exigir int/float, tipos booleanos y estructurados también anidados, update y merge, error/recovery, override/delete, scope y update de tipo incompatible. |
| LGM-01–05 | Creación Python/REST/nombre automático, params/tags, metric identity sync/async/múltiples runs/datasets, lifecycle, experimento inactivo/ausente y entradas duplicadas REST. |
| REG-01–07 | CRUD/descripciones/rename/tags/stages/latest/aliases, lectura HTTP y Mongo; esquema de tags key/value obligatorio en registry, separado del lifecycle. Tracking conserva k/v. |
| REG-08–11 | Copy real desde runs:/ pasando por models:/, descarga de archivo local, redacción, eliminación de hijos, conflictos y concurrencia pequeña. |
| PRM-01–04 | Metadata donde exista API upstream, versiones/templates/tags/aliases, render desde proceso nuevo, separación modelo/prompt, filtros/paginación sin orden explícito y búsqueda de versiones desde 3.11.1. |
| LNK-01–02 | Vínculos prompt/run, prompt/logged model, prompt/trace y trace/run, idempotencia, lectura HTTP y Mongo, búsquedas de trace por prompt/run. |
| Búsquedas | Experimentos, runs, traces, logged models, registered models, model versions, prompts y placeholder evaluation datasets. Filtros, AND, casos vacíos/no coincidencia, orden/paginación e inputs inválidos por familia. |
| QTM-01–04 | Counts en las tres vistas, filtros/dimensiones, latencia AVG/percentil, buckets UTC y ns→ms, token SUM, session COUNT y cost SUM según upstream. Se comprueba el límite, sin prometer paginación de grupos que el store actual no ofrece. |

Extensiones de profundidad del inventario que siguen pendientes:

- Exhaustividad combinatoria de filtros/orderings: todas las variantes literales y aliases
  de timestamps, claves con dólares/espacios, empates exactos y tokens reutilizados con otra query.
- Búsquedas de logged models por nombre de dataset sin digest: los escenarios actuales usan
  digests explícitos; el handler HTTP 3.5.1 convierte el digest omitido en un filtro nulo.
- Filtros de timestamps de model versions cuando el validador upstream no los admite:
  el caso se informa como `UNAVAILABLE_UPSTREAM`, no como cobertura aprobada.
- IssueReference y cada combinación de errores/campos inmutables de assessments.
- Consultas multiexperimento y todos los límites temporales; costos/autoritatividad anidados
  con todas las variantes upstream. Dimensiones modelo/proveedor, agregaciones numéricas con
  overrides y agrupación de valores estructurados sí tienen escenarios.
- Vínculo de trace creado dentro de un run mediante metadata de origen; errores de vínculo y
  pruebas discriminantes que exhiban los filtros avanzados ignorados actualmente.
- Copia dentro del mismo modelo y cadenas de copias. Stages con/sin archivar, alias latest,
  lifecycle básico y copia real entre modelos sí están cubiertos.
- Todos los casos negativos de combinaciones de delete-traces y mutaciones sobre recursos borrados.
- S3 y proxy de artifacts con los proveedores existentes de MLflow; matriz de lectores/escritores
  de distinta versión. La primera matriz utiliza almacenamiento local y cliente/servidor iguales.

Las limitaciones de producto explícitas aparecen como `NOT_IMPLEMENTED`; no se cuentan como
casos aprobados. Esta carpeta no arregla el producto para cambiar su resultado.
