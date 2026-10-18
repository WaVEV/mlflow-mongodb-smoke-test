# Plan de smoke e integración por versión de MLflow

Este documento es un inventario de alcance, no un reporte de pruebas ejecutadas.
La implementación y sus extensiones pendientes están mapeadas en [COVERAGE.md](COVERAGE.md).
El objetivo es probar el plugin instalado mediante un servidor real, desde MLflow 3.5.0
hasta la última publicación estable de MLflow 3.x. Cliente y servidor usan la misma versión
en cada ejecución. La compatibilidad entre versiones distintas queda para otra matriz.

## Ejecución y criterios

- Descubrir versiones estables no retiradas, incluyendo cada parche desde 3.5.0.
- Crear un entorno por versión e instalar el código de trabajo del plugin. Registrar versión
  de Python, MLflow, dependencias y huella del código probado, incluidos cambios sin commit.
- Usar MongoDB 8.0+ existente; Docker no es necesario. El runner genera nombres de bases
  exclusivos y solo puede limpiar bases que él mismo creó durante esa ejecución.
- Cada escenario independiente comienza con una base nueva y un servidor nuevo. Las acciones
  de un mismo lifecycle conservan sus datos hasta terminar ese escenario. Esto evita también
  cachés de servidor heredadas de otro escenario.
- Mantener artifacts y directorio de trabajo separados por escenario. Empezar con el proveedor
  local existente de MLflow; S3 y el proxy de artifacts serán configuraciones adicionales.
- Probar llamadas del cliente Python y peticiones REST explícitas. Las escrituras deben pasar
  por el servidor; Mongo se consulta para verificar, no para preparar artificialmente resultados.
- Tras cada escritura comprobar el estado esperado por HTTP y por Mongo. Un HTTP 200 por sí
  solo no acredita que se guardó nada. Para borrados comprobar ausencia o estado lógico según
  el contrato de la entidad. No exigir eliminación física cuando corresponde borrado lógico.
- Las lecturas y búsquedas deben comparar contenido, IDs, cardinalidad y orden contra datos
  conocidos del escenario; no derivar el resultado esperado del parser o repositorio probado.
- Usar esperas acotadas para exportación y logging asíncronos. Distinguir timeout, error de
  instalación, error al arrancar servidor y fallo de una comprobación.
- Registrar operación, ruta HTTP, consulta Mongo con datos ficticios, resultado esperado/obtenido,
  logs del cliente/servidor y duración. No registrar credenciales de conexión.
- Continuar con escenarios independientes tras un fallo. Dentro de un escenario indicar cuáles
  pasos quedaron bloqueados por un fallo previo, en lugar de contarlos como pasados.
- Apagar procesos y limpiar bases generadas también al fallar o interrumpirse. Los reportes y
  logs sobreviven a la limpieza. La inspección opcional de una base fallida requiere un modo
  explícito de conservación.

## Estados del reporte

- PASSED: se ejecutaron las comprobaciones esperadas y coincidieron.
- FAILED: una operación requerida no cumple el resultado esperado, aunque sea un defecto conocido.
- UNAVAILABLE_UPSTREAM: esa versión de MLflow no ofrece la API o sintaxis; indicar evidencia.
- NOT_IMPLEMENTED: función fuera del soporte actual del plugin; visible en el inventario, sin
  contarla como prueba exitosa ni como incompatibilidad de MLflow.
- BLOCKED / SETUP_FAILED: el escenario no pudo ejecutarse por una dependencia fallida o por entorno.

No convertir automáticamente excepciones, respuestas 404/500 o resultados vacíos en skips.
Resolver las capacidades esperadas antes de ejecutar cada caso. Para APIs que cambian de ruta,
mapear el método público y el endpoint de la versión instalada antes de implementar el caso.

## Experimentos

- EXP-01: crear por MlflowClient y por REST explícito, con nombre, artifact location y tags.
  Obtener por ID y por nombre; comprobar identidad, tiempos, lifecycle y valores persistidos.
- EXP-02: crear mediante mlflow.set_experiment y repetir la selección del mismo nombre;
  comprobar que no aparece otro experimento.
- EXP-03: renombrar; conservar ID, tags y artifact location; resolver el nombre nuevo y dejar
  de resolver el anterior. Incluir renombrar al mismo nombre.
- EXP-04: crear un tag después del experimento mediante set_experiment_tag. Comprobarlo por
  get HTTP, por búsqueda del tag y en el documento Mongo.
- EXP-05: modificar el valor de un tag existente mediante set_experiment_tag. Comprobar el
  valor nuevo, ausencia de duplicados y que el filtro por el valor anterior ya no lo encuentre.
- EXP-06: borrar un tag mediante delete_experiment_tag. Comprobar ausencia por HTTP y Mongo,
  que no coincida el filtro anterior y que los otros tags permanezcan.
- EXP-07: borrar/restaurar el experimento; validar vistas activo/borrado/todos, conservación
  de metadatos y el comportamiento actual de sus runs, sin asumir un borrado en cascada.
- EXP-08: nombres duplicados activos o borrados, rename en conflicto, nombre inválido,
  IDs inexistentes y mutaciones prohibidas en un experimento borrado. Comprobar códigos públicos
  y ausencia de escrituras no deseadas.

EXP-04, EXP-05 y EXP-06 son requisitos explícitos: deben reportarse como FAILED cuando
la escritura no persista, no omitirse ni marcarse xfail. Las mutaciones se implementaron
posteriormente en la rama de experimentos como un cambio separado del smoke; falta verificar
su ejecución. No modificar el producto desde el smoke para hacer pasar estos escenarios.

## Runs, logging e inputs/outputs

- RUN-01: crear por cliente y REST; obtener y validar experimento, usuario, nombre, tags,
  estado inicial, tiempos y artifact URI. Incluir nombre automático y mlflow.runName.
- RUN-02: actualizar estado, end time y nombre; comprobar sincronización del nombre y su tag.
- RUN-03: logging individual y en batch de params, métricas y tags. Modificar tags y repetir
  un param idéntico; rechazar un valor distinto para un param inmutable.
- RUN-04: logging asíncrono individual/batch y espera explícita. Confirmar persistencia y
  propagación de errores al esperar una operación fallida.
- RUN-05: historial de métricas con varios steps/timestamps, entrega fuera de orden, empates
  y paginación. Separar la métrica vigente del historial; incluir NaN e infinitos con el transporte
  admitido por MLflow y validar su representación sin asumir igualdad ordinaria de NaN.
- RUN-06: dataset inputs con name, digest, source, schema, profile y tags de contexto. Repetir
  el input y comprobar identidad, preservación de tags y arrays persistidos k/v anidados.
- RUN-07: model inputs y outputs; verificar IDs, steps, orden y el contrato de duplicados.
- RUN-08: borrar/restaurar; preservar datos e historial, comprobar deleted_time y repetición
  de transiciones. Rechazar escrituras que requieren lifecycle activo.
- RUN-09: run inexistente, experimento inexistente/borrado, batch conflictivo y batch vacío;
  comprobar código público y que las validaciones previas no produzcan escrituras parciales.
  No prometer atomicidad general entre colecciones que el plugin no ofrece.

## Traces y spans

- TRC-01: adaptar trace_test.py y test.py: span simple y función decorada, inputs/outputs,
  estado OK, nombre e ID; esperar exportación y recuperar la traza completa.
- TRC-02: adaptar trace_test2.py: spans padre/hijo, relación de IDs y payload de cada span.
- TRC-03: adaptar test3.py: atributos string, numéricos y estructurados, inputs/outputs;
  comparar contenido recuperado con spans persistidos, no solo presencia de documentos.
- TRC-04: adaptar test4.py: excepción intencional, estado ERROR y evento de excepción,
  sin convertir la excepción esperada del escenario en fallo del runner.
- TRC-05: adaptar test5.py: dos experimentos preparados en el propio escenario, con trazas
  independientes; búsqueda y recuperación no deben mezclar sus resultados.
- TRC-06: crear metadatos mediante API disponible; comprobar tags, metadata, previews,
  client_request_id, tiempos y tamaño cuando esos campos formen parte de la API instalada.
- TRC-07: logging de spans sync/async, reentrega idempotente y recepción de un hijo antes
  de la raíz por las rutas que ofrezca esa versión. Verificar resumen final sin perder payload.
- TRC-08: get_trace_info y get_trace; lecturas batch donde exista endpoint, orden de respuesta,
  IDs ausentes, alcance por experimento y omisión de trazas incompletas según contrato.
- TRC-09: set/delete de tags de traza, reemplazo sin duplicados, valores literales y protección
  de ubicación de spans/artifacts. Tag o traza inexistentes deben dar el error previsto.
- TRC-10: borrar por IDs y por timestamp con límite; comprobar aislamiento por experimento
  y cascada de documentos de spans/assessments. Verificar combinaciones inválidas de parámetros.
  No asumir que este borrado elimina también los archivos del proveedor de artifacts.
- TRC-11: token usage, costos y metadata de sesión/usuario cuando haya helpers upstream;
  evitar doble conteo de spans anidados y respetar metadata autoritativa de la traza.

### Recuperación y artifacts

- ART-01: en MLflow 3.5 recuperar por el cliente la traza con todos sus spans desde artifacts;
  comprobar URI anunciada, existencia del payload y contenido, además de las copias en Mongo.
- ART-02: en versiones que anuncian TRACKING_STORE, comprobar recuperación completa por
  las rutas de servidor correspondientes y coherencia con Mongo.
- ART-03: reiniciar el servidor con la misma base y artifact root dentro del escenario y leer
  de nuevo. Un nuevo proceso cliente evita que una caché local oculte falta de persistencia.
- No sustituir el proveedor de artifacts de MLflow ni implementar uno propio. No afirmar
  compatibilidad entre un lector 3.5 y un escritor moderno con esta matriz de misma versión.

## Assessments

- ASM-01: crear/obtener Feedback y Expectation, tanto valores escalares como JSON estructurado;
  comprobar valores nativos y su incorporación a get_trace_info/get_trace.
- ASM-02: actualizar valor, nombre, rationale y metadata; conservar campos inmutables y
  comprobar el merge previsto de metadata.
- ASM-03: establecer y limpiar AssessmentError, conservando código y mensaje.
- ASM-04: override de un assessment; invalidación del original y restauración al borrar
  el reemplazo. Borrado repetido y conservación de otros assessments.
- ASM-05: IDs ausentes, lookup con otra traza y tipos incompatibles en updates; errores
  públicos y ausencia de modificaciones ajenas. IssueReference solo cuando exista upstream.

## Logged models

- LGM-01: crear/obtener con y sin nombre, source_run_id, model_type, params, tags y status
  PENDING; comprobar artifact location y rechazo de experimento ausente/borrado.
- LGM-02: set/delete de tags, sustitución, lotes vacíos, claves literales y aislamiento entre modelos.
- LGM-03: métricas asociadas mediante log_batch/log_metric con model_id, run_id y dataset;
  comprobar historial de varios runs y datasets, incluyendo logging asíncrono.
- LGM-04: borrar; ocultar en búsquedas, conservar datos según lifecycle y comprobar acceso
  a borrados solamente por rutas que expongan esa opción. Repetir el borrado según contrato.
- LGM-05: tags/params de creación duplicados o nulos, modelo inexistente y operaciones sobre
  borrados; errores públicos y ausencia de creación parcial.
- La finalización de status de logged models no está implementada. No usar log_model de un
  flavor como supuesto caso exitoso si requiere finalize_logged_model. Mostrar esta limitación
  en el inventario; no es un fallo introducido por soportar MLflow 3.5.

## Registered models, versiones y aliases

- REG-01: crear/obtener/actualizar descripción; comprobar tags iniciales, unicidad y timestamps.
- REG-02: renombrar; versiones y aliases conservan identidad, y el nombre anterior deja de resolver.
- REG-03: set/delete de tags del modelo y de la versión; reemplazo sin duplicados, claves literales
  y aislamiento entre recursos. Contrastar el formato realmente persistido sin introducir migraciones.
- REG-04: crear varias versiones; numeración creciente, source, run_id, run_link, descripción,
  estado READY y download URI. Una URI s3 ficticia solo prueba metadata, no acceso real a S3.
- REG-05: actualizar descripción de versión y consultar latest_versions por stage.
- REG-06: transiciones de stage con y sin archive_existing_versions; comprobar versiones vecinas.
- REG-07: asignar/reasignar/resolver/borrar aliases; reflejo en modelo y versiones, alias latest
  sintético cuando corresponda y rechazo de aliases reservados o destinos inexistentes.
- REG-08: copiar versión entre modelos por la API heredada; comprobar metadata y resolución
  de sources de tipo models:/ y runs:/ cuando se preparen artifacts reales para ese caso.
- REG-09: borrar una versión; comprobar ocultación, redacción persistida y limpieza de aliases.
  Borrar el modelo padre y verificar eliminación de hijos según el contrato actual.
- REG-10: recursos inexistentes, nombres duplicados, rename en conflicto, versiones/stages
  inválidos y operaciones posteriores a borrar; comprobar códigos y no mutación de otros objetos.
- REG-11: creación concurrente pequeña de versiones para detectar colisiones o números repetidos;
  no presentarla como prueba de carga o garantía de transacciones.

## Prompts y vínculos

- PRM-01: crear/obtener/borrar prompt, descripción y tags; comprobar la representación en registry.
- PRM-02: registrar dos versiones con templates, commit message y tags; leer por número y alias,
  set/delete de tags y reasignación/borrado de aliases.
- PRM-03: borrar versión sin afectar la otra; borrar prompt y comprobar hijos. Rechazar colisiones
  de nombre entre prompts y modelos y comprobar separación de sus búsquedas.
- PRM-04: cargar/renderizar un template con variables por API pública cuando esté disponible;
  el resultado debe corresponder a la versión persistida, sin depender de caché de cliente.
- LNK-01: vincular prompts a traces, runs y logged models mediante las APIs heredadas disponibles.
  Leer ambos lados que la API exponga y consultar referencias/tags Mongo; repetir sin duplicar.
- LNK-02: vincular traces a runs; comprobar idempotencia y búsquedas por run_id, tanto vínculos
  explícitos como source_run metadata. No presentar unlink_traces_from_run como implementado.
- Para vínculos heredados, comprobar qué parte corre en el cliente y qué llamada llega al servidor;
  no confundir un método presente en AbstractStore con soporte efectivo por HTTP.

## Inventario de búsquedas

Cada familia incluye página vacía, coincidencias y no coincidencias, AND, aislamiento de recursos,
orden ASC/DESC, campos ausentes, empates y paginación sin omisiones/duplicados. Comprobar filtros
antes de paginar y resultados completos, no solo IDs. No exigir ausencia de token en una página
llena si el contrato permite una página vacía final; sí exigir terminación.

| Familia | Casos que debe transportar el smoke al servidor |
| --- | --- |
| Experimentos | Nombre =/!=/LIKE/ILIKE, tags, case sensitivity, timestamps numéricos, vistas de lifecycle, nombre actualizado y existencia de tags cuando lo acepte MLflow. |
| Runs | Métricas con comparadores numéricos y resumen vigente frente a historial; params/tags =/!=/LIKE/ILIKE y existencia disponible; atributos de nombre/estado/tiempos; run_id IN/NOT IN; datasets name/digest/context y sus combinaciones. |
| Traces | Estado, nombre, timestamp/duración/end time disponibles, client_request_id disponible, tags y request_metadata, LIKE/ILIKE/RLIKE disponibles, run_id, prompt name/version; orden por atributos/tags/metadata permitidos. |
| Logged models | Atributos y aliases de timestamps, model_id IN, params/tags y pertenencia, métricas contra cualquier fila de historial, scope por dataset name/digest; ordenar por métrica con scope independiente del filtro, nulos al final y lifecycle. |
| Registered models | Nombre y tags con comparadores admitidos; timestamps para ordenar, latest_versions en resultados, exclusión de borrados y separación de prompts. |
| Model versions | Nombre/run_id con IN, source_path, versión y timestamps numéricos cuando el validador upstream los admite, tags; orden y paginación estables, exclusión de borrados y prompts salvo filtro explícito. |
| Prompts | Búsqueda de prompts por nombre/tags; versiones por filtros, orden y páginas cuando exista esa API, actualmente cubierta por los tests a partir de 3.11.1. |
| Evaluation datasets | search_datasets es un placeholder: validar su respuesta vacía y errores de consulta como contrato limitado, sin declararlo búsqueda funcional de datasets persistidos. Es distinto de buscar runs por sus dataset inputs. |

Casos transversales: claves con puntos/dólares/espacios, valores vacíos, porcentajes/guiones bajos
en patrones, comparación entre claves distintas y condiciones repetidas de una misma clave.
Incluir operandos inválidos, regex inválidas, orden duplicado/no admitido, tamaño de página inválido
y tokens inválidos o incompatibles con la consulta donde el contrato lo valide.

Los filtros de traces por span/feedback/expectation/issue se omiten actualmente en el parser del
plugin mediante continue; model_id tampoco se aplica. No contarlos como búsquedas soportadas ni
como éxitos por devolver HTTP 200. Mostrar limitación conocida y, en un bloque de diagnóstico,
consultas que deberían discriminar resultados para hacer visible ese comportamiento incompleto.

## Métricas agregadas de traces

- QTM-01: desde la API upstream disponible en 3.9, vistas TRACES/SPANS/ASSESSMENTS; counts,
  latencia, tokens y valores de assessments, con agregaciones admitidas (incluido percentil).
- QTM-02: dimensiones, filtros, alcance por experimento, ventanas y buckets UTC; comprobar
  conversión de nanosegundos/milisegundos y límites temporales con datos controlados por API.
- QTM-03: bool/números/strings/estructuras en assessments; respetar reglas de conversión y exclusión.
- QTM-04: sesiones, costos y dimensiones de modelo/proveedor solo si MLflow ofrece la capacidad.
- El resultado esperado se calcula a partir de los datos conocidos; Mongo verifica los registros
  fuente. No ejecutar el pipeline del repositorio como oráculo del mismo endpoint.
- En versiones previas reportar API no disponible; no confundir con logging de métricas de runs.

## Límites y cobertura posterior

- No incorporar implementaciones del plugin para hacer pasar estos escenarios. Registrar defectos.
- No validar webhook delivery, workspace provider, proveedor de artifacts propio, archive de trazas,
  unlink de trazas ni CRUD de evaluation datasets como funciones implementadas de V1.
- Los tests de fallos de driver, cursores parciales, límites BSON y transacciones necesitan otra
  estrategia; no sustituir esos tests con este smoke ni simularlos con escrituras directas a Mongo.
- Antes de codificar, completar el mapeo por versión entre cada operación, método Python, ruta REST,
  colecciones/documentos y assertions. Variantes con la misma estructura podrán parametrizarse;
  lifecycles con comportamiento distinto permanecerán separados.
