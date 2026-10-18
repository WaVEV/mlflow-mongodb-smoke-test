# Decisiones técnicas

- El smoke vive separado del plugin y prueba un checkout local explícito mediante `--plugin-dir`.
- El runner instala el checkout en modo editable en un entorno temporal por versión de MLflow.
- La huella de fuentes incluye el código del smoke y el checkout del plugin para detectar cambios durante una ejecución.
- Los logs, reportes, entornos y credenciales no forman parte del repositorio.
