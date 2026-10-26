# Matriz de cumplimiento de Actividad 6

**Alcance:** instrucciones de Módulo 2 / Semana 7. Esta tabla verifica entregables concretos publicados; no es calificación oficial. **Formato de entrega confirmado por el estudiante:** enlace al repositorio GitHub en sustitución del ZIP.

| Instrucción / criterio | Archivo / evidencia reproducible | Estado |
|---|---|---|
| Problema financiero y objetivo SMART | README §§1–2; docs/protocolo.md; PDF p. 1–2 | Cumple |
| Código de al menos dos modelos | fuentes/train.py: LogisticRegression y RandomForest | Cumple |
| Baseline | resultados/baseline_dummy.json | Cumple |
| Matriz de confusión e interpretación | resultados/comparacion_modelos.csv; resultados/figuras/03_matrices_confusion.png; PDF p. 3 | Cumple |
| Exactitud, precisión, recall, F1 | resultados/comparacion_modelos.csv; notebook bloque 3 | Cumple |
| Ajuste de umbral validación | resultados/umbrales_LogisticRegression.csv y umbrales_RandomForest.csv; seleccion_congelada.json | Cumple |
| Curva ROC y ROC-AUC | resultados/figuras/01_02_curvas_roc_pr.png; PDF p. 4 | Cumple |
| Validación cruzada | resultados/gridsearch_cv.csv; PDF p. 5; protocolo | Cumple |
| Comparación con baseline | resultados/baseline_dummy.json; PDF p. 5 | Cumple |
| Implementación de A/B | resultados/simulacion_ab.json; figura 05; notebook bloque 7; PDF p. 6 | Cumple como SIMULACIÓN offline, no ensayo causal |
| Significancia estadística | resultados/mcnemar.json; bootstrap_pareado.csv; PDF p. 5–6 | Cumple con limitaciones de diseño |
| Justificación técnica y de impacto comercial | README §6; PDF p. 3 y 7; Trello tarjeta de decisión | Cumple, escenario de costos reales pendiente de datos |
| Evidencia de experimentación | GitHub Actions en verde, CSV, figuras, notebook ejecutado 8/8 | Cumple |
| Reporte técnico 6–8 páginas | docs/Reporte_Tecnico_Actividad6.pdf (7 páginas) | Cumple |
| Tablero Trello visual | https://trello.com/b/EIKcvZ7j/banking-conversion-ai-actividad-6 (5 listas, 11 tarjetas) | Construido, acceso docente pendiente |
| Entrega en ZIP | Enlace GitHub en sustitución, confirmado por el estudiante | Cumple formato acordado |

## Hallazgos críticos que NO deben ocultarse
1. RandomForest ofrece mejor F1, AP y ROC-AUC; LogisticRegression alcanza mayor recall test y detecta 76 respuestas positivas adicionales. La selección fue sobre validation, nunca test.
2. McNemar contrasta errores discordantes en observaciones emparejadas, pero no constituye A/B y no mide ventas causadas por IA.
3. La simulación A/B selecciona registros según su score y **usa desenlaces históricos observados**; su p exploratorio no es evidencia de efecto causal de la política. El diseño de un ensayo aleatorizado real figura en docs/protocolo.md y PDF p. 6.
4. El dataset carece de ID de cliente, impide agrupar por persona y no demuestra generalización prospectiva.
5. La documentación y las cifras se generan automáticamente y no son etiquetas producidas manualmente.

## Integridad
Recalcular métricas desde `resultados/predicciones_test.csv`; contrastar errores McNemar y delta F1 bootstrap; comprobar los SHA256 en `resultados/fuente_y_calidad.json`. El script y las pruebas permiten reproducir la entrega sin token ni cuenta privada de MLflow.

## Mejora de rendimiento y contexto de negocio — análisis EXPLORATORIO

Se incorporó un tercer modelo HistGradientBoosting y una comparación de F1 con umbrales elegidos exclusivamente sobre validation con el mismo protocolo para los tres modelos. La nueva comparación de F1 en test es **exploratoria**, no una prueba ciega adicional, porque la prueba ya se había inspeccionado.

- F1 homogéneo test: LR 0.3662; RF **0.4117**; HGB 0.3990.
- RF con umbral F1 0.19: precision 41.71%, recall 40.64%; contrasta con F2 original, precision 25.61%, recall 57.56%.
- Baseline siempre positivo: F1 0.2095; evita comparar únicamente con un DummyClassifier de F1 nulo.
- Capacidad del 20%: Precision@20% LR 28.32%, RF 30.20%, HGB 30.25%; lift 2.42x, 2.58x, 2.59x, respectivamente; **ranking descriptivo histórico, no causal**.
- Evidencia: [anexo PDF](Anexo_Mejora_Modelos_Actividad6.pdf), [CSV F1](../resultados/analisis_adicional/comparacion_f1_umbral_validado.csv), [figura lift](../resultados/figuras/07_lift_presupuesto20.png), [scripts](../fuentes/analisis_adicional.py).

### Capturas auténticas de Trello
El PDF principal contiene 7 páginas sin capturas del tablero mientras no se suban los dos PNG a [evidencias/trello](../evidencias/trello/). Ya se programó su inclusión automática como octava página: no se deben presentar imágenes fabricadas como pantallas reales.
