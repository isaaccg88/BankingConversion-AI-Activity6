# Banking Conversion AI — Actividad 6

**Gestión de Proyectos de Inteligencia Artificial · Módulo 2 / Semana 7**  
**Contexto:** financiero — contratación de depósitos a plazo mediante campañas telefónicas.  
**Repositorio:** https://github.com/isaaccg88/BankingConversion-AI-Activity6  
**Tablero auténtico de Trello:** https://trello.com/b/EIKcvZ7j/banking-conversion-ai-actividad-6

> **Estado: EXPERIMENTO COMPLETADO Y VALIDADO.** GitHub Actions ejecutó entrenamiento, CV, selección de umbrales, test y comparaciones estadísticas; publicó resultados, notebook ejecutado (8/8 celdas sin errores) y PDF verificado de **7 páginas**. Los datos proceden de UCI; ninguna tasa ni ganancia es inventada.

## 1. Problema y objetivo SMART

Una entidad bancaria debe elegir a qué clientes contactar cuando su capacidad comercial es limitada. El objetivo es evaluar en un conjunto público **si un clasificador permite priorizar mejor las observaciones con contratación histórica que una selección aleatoria**, y si supera a modelos alternativos en métricas estadísticas.

**Objetivo SMART académico:** usando UCI Bank Marketing (45,211 registros), comparar **regresión logística y Random Forest contra DummyClassifier** mediante cinco folds estratificados, curvas ROC y Precision-Recall, matriz de confusión y un test de 20% reservado; seleccionar umbrales **solo con validación** y evaluar una política de contacto del 20% de capacidad mediante una simulación A/B offline. Documentar significancia e intervalos y recomendar un modelo bajo restricciones declaradas, **en una ejecución reproducible del proyecto**. Las metas numéricas se interpretan sobre los datos reales, no se fijan arbitrariamente para garantizar éxito.

### Alcance y límites
El dataset es **histórico y observacional**. Describe el resultado de campañas que ya ocurrieron, no los resultados contrafactuales de contactar o no contactar. La simulación A/B compara estrategias de selección **offline**; no prueba que el modelo cause conversiones adicionales, ni constituye un ensayo aleatorizado real. El desempeño temporal futuro y la generalización a otros bancos no están validados.

## 2. Datos, trazabilidad y prevención de fuga de información

**Fuente oficial:** [UCI Bank Marketing, ID 222](https://archive.ics.uci.edu/dataset/222/bank+marketing), **DOI [10.24432/C5K306](https://doi.org/10.24432/C5K306)**, licencia CC BY 4.0 (Moro, Rita y Cortez). Usamos **`bank-full.csv`**, versión histórica de 45,211 observaciones y 16 predictoras originales más el objetivo `y` (`yes/no`).

La fuente advierte que `duration` es la duración de una llamada y no puede conocerse antes de realizarla. Por ello se elimina **obligatoriamente**. En el protocolo conservador de predicción *antes del contacto* también se excluyen `contact`, `day`, `month` y `campaign`, porque su disponibilidad o definición pueden depender de la campaña en curso. Quedan **11 predictoras**: `age`, `balance`, `pdays`, `previous`, `job`, `marital`, `education`, `default`, `housing`, `loan` y `poutcome`.

**Control importante:** no existen identificadores de cliente que garanticen la independencia de registros de la misma persona. Las particiones estratificadas aleatorias reproducibles son una elección académica, no una evaluación prospectiva por fecha o por cliente. `pdays=-1` y `unknown` mantienen su codificación de origen y no se interpretan automáticamente como nulos.

El código conserva el CSV oficial con SHA256 y un manifiesto de calidad en `resultados/fuente_y_calidad.json`. Las particiones train / validación / test se fijan con semilla 42 en `datos/splits.csv`. Se ajustan imputación, escalado y One-Hot Encoding **dentro del Pipeline de cada fold**; por tanto, no se ajustan transformaciones en test.

## 3. Protocolo de evaluación

| Elemento | Decisión preestablecida |
|---|---|
| Clasificadores principales | LogisticRegression, RandomForestClassifier |
| Línea base | DummyClassifier con probabilidad de clase |
| Particiones | Entrenamiento 60%, validación 20%, test 20%; estratificadas |
| Validación cruzada | StratifiedKFold, 5 folds, semilla 42 |
| Selección de hiperparámetros | GridSearchCV, maximiza Average Precision solo en train |
| Umbral de cada modelo | Maximiza F2 en validation con precision mínima 0.20 si es factible |
| Selección del modelo | Mayor F2 de validation, desempate Average Precision de CV |
| Métricas | Accuracy, balanced accuracy, precision, recall, F1, F2, especificidad, ROC-AUC, Average Precision |
| Comparación estadística | Test exacto McNemar de errores discordantes y bootstrap pareado de F1 (1,000 iteraciones) |
| Simulación A/B offline | División aleatoria de test en dos grupos disjuntos; política A aleatoria y B ranking RF bajo presupuesto de llamadas del 20% |
| Contraste A/B | Diferencia de tasas históricas, IC95 de proporciones y p exploratorio; **sin lectura causal** |
| Negocio | Ganancias históricas por contacto seleccionado bajo presupuesto fijo, sin asignar ingresos monetarios inventados |

**Dos preguntas distintas:** el contraste pareado compara clasificadores sobre *las mismas observaciones*; la simulación A/B reparte observaciones en *grupos disjuntos* y compara estrategias de selección. Las dos son retrospectivas: solo una prueba de campo aleatorizada con resultados prospectivos permitiría inferencia causal.

## 4. Organización del repositorio

| Ruta | Contenido |
|---|---|
| [fuentes/datos.py](fuentes/datos.py) | Descarga de UCI, validación, SHA256, dataset y splits |
| [fuentes/evaluation.py](fuentes/evaluation.py) | Métricas, matrices y selección de umbral |
| [fuentes/train.py](fuentes/train.py) | Entrenamiento, CV, ROC/PR, estadística y A/B offline |
| [tests/test_evaluation.py](tests/test_evaluation.py) | Verificaciones unitarias |
| [datos/](datos/) | CSV oficial y particiones reproducibles; se generan |
| [resultados/](resultados/) | Comparación real, predicciones, tablas, seis figuras y contrastes |
| [notebooks/Evaluacion_Actividad6.ipynb](notebooks/Evaluacion_Actividad6.ipynb) | Notebook con revisión ejecutada de los resultados generados |
| [docs/Reporte_Tecnico_Actividad6.pdf](docs/Reporte_Tecnico_Actividad6.pdf) | Reporte académico reproducible de 6–8 páginas (se genera) |
| [docs/protocolo.md](docs/protocolo.md) | Diseño experimental, límites y criterios de evaluación |
| [.github/workflows/](.github/workflows/) | Ejecución, verificación y publicación automatizadas |
| [CHANGELOG.md](CHANGELOG.md) | Versiones y decisiones |

## 5. Reproducir localmente

Requiere Python 3.11 compatible; también puede instalarse en un entorno virtual separado de la instalación de MLflow de la Actividad 5.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m fuentes.train
```

No se requieren tokens de Hugging Face ni MLflow para la Actividad 6. En GitHub, el workflow **Actividad 6 - Entrenamiento y evidencia real** realiza estos pasos y publica sus resultados después de comprobar que son válidos.

## 6. Resultados reales, significancia y conclusiones

**Las cifras siguientes fueron calculadas a partir de las 9,043 observaciones de prueba.** El modelo se seleccionó exclusivamente mediante validación, no por inspección retrospectiva de las métricas de test.

| Indicador de test | LogisticRegression | RandomForest |
|---|---:|---:|
| AP media de 5 folds de entrenamiento | 0.3435 | **0.3670** |
| F2 en validación | 0.4640 | **0.4809** |
| Umbral congelado | 0.11 | 0.12 |
| Accuracy | 0.6666 | **0.7547** |
| Precision | 0.2059 | **0.2561** |
| Recall | **0.6474** | 0.5756 |
| F1-score | 0.3124 | **0.3545** |
| ROC-AUC | 0.7204 | **0.7392** |
| Average Precision | 0.3470 | **0.3838** |
| Verdaderos positivos (TP) | **685** | 609 |
| Falsos negativos (FN) | **373** | 449 |
| Falsos positivos (FP) | 2,642 | **1,769** |
| Verdaderos negativos (TN) | 5,343 | **6,216** |

**Interpretación:** Random Forest reduce contactos improductivos (873 falsos positivos menos) y aumenta precision y F1, pero deja sin detectar 76 conversiones históricas adicionales con respecto a la regresión logística. Por tanto, la elección depende de la restricción de llamadas y del costo de las oportunidades perdidas. La selección formal fue RandomForest por mayor F2 de validación.

**Baseline:** el modelo mayoritario DummyClassifier logra una accuracy elevada simplemente por la prevalencia de no contratación, pero tiene recall=0 y F1=0.

**Significancia predictiva:** McNemar exacto sobre discordancias de error: b=216 y c=1,013; p=1.12×10⁻¹²³. Bootstrap pareado de 1,000 remuestreos: diferencia F1(RF−LR)=**0.0421**, IC95 **[0.0313, 0.0526]**. Estos contrastes respaldan una diferencia observada entre predictores en esta muestra; no prueban generalización temporal ni impacto económico.

**Simulación A/B retrospectiva:** con 904 seleccionados por estrategia, A (aleatoria) identificó 88 conversiones históricas (9.73%) y B (ranking RF) 270 (29.87%). Diferencia descriptiva: 20.13 puntos porcentuales (IC95 aproximado 16.58 a 23.69 pp). **No es una prueba A/B de campo ni demuestra incremento causal por utilizar IA.** El protocolo explica por qué la comparación observacional está condicionada por la selección.

**Figuras trazables:** [curvas ROC y PR](resultados/figuras/01_02_curvas_roc_pr.png), [matrices de confusión](resultados/figuras/03_matrices_confusion.png), [hiperparámetros CV](resultados/figuras/04_cv.png), [A/B offline](resultados/figuras/05_ab_offline.png), [F1 test](resultados/figuras/06_f1_test.png).

**Documentos ejecutados:** [notebook validado](notebooks/Evaluacion_Actividad6.ipynb) y [reporte PDF académico de 7 páginas](docs/Reporte_Tecnico_Actividad6.pdf).

## 6.1. Ampliación exploratoria para evaluar el F1

Se detectó que el F1 original de RandomForest (0.3545) era moderado. Se preservó el protocolo original orientado a F2 y se ejecutó un experimento COMPLEMENTARIO, nunca presentado como validación externa: **la prueba original ya había sido inspeccionada**. Este hecho se registra explícitamente en el [resumen del análisis adicional](resultados/analisis_adicional/resumen_adicional.json).

El nuevo procedimiento entrenó **HistGradientBoosting** (cinco folds, selección de hiperparámetros por Average Precision), y para los TRES algoritmos seleccionó umbrales maximizando **F1 solo en validation**, con precision mínima de 0.20.

| Modelo | Umbral seleccionado en validación | F1 en validación | F1 en test exploratorio | Precisión test | Recall test |
|---|---:|---:|---:|---:|---:|
| LogisticRegression | 0.17 | 0.3739 | 0.3662 | 34.12% | 39.51% |
| RandomForest | **0.19** | 0.4278 | **0.4117** | **41.71%** | 40.64% |
| HistGradientBoosting | 0.21 | **0.4373** | 0.3990 | 41.89% | 38.09% |

**Interpretación:** al favorecer F1, RandomForest mejora su precisión del 25.61% al 41.71%, pero reduce su recall del 57.56% al 40.64%: **el umbral comercial involucra un trade-off real**. Un baseline trivial que predice todos los clientes positivos tiene F1=0.2095, de modo que el resultado ofrece capacidad predictiva superior a ese referente, sin implicar que esté listo para despliegue real.

Para cuantificar el interés comercial sin modificar el conjunto original, evaluamos **Precision@20% y lift** al contactar solo 1,808 de los 9,043 registros test con puntuación más alta:

| Modelo | Contactos seleccionados | Conversiones históricas entre seleccionados | Precision@20% | Lift vs prevalencia |
|---|---:|---:|---:|---:|
| LogisticRegression | 1,808 | 512 | 28.32% | 2.42× |
| RandomForest | 1,808 | 546 | 30.20% | 2.58× |
| HistGradientBoosting | 1,808 | 547 | 30.25% | 2.59× |

La diferencia de una conversión histórica entre RF e HistGradientBoosting en top-20% **no demuestra ventaja material ni estadística**. Ni estas métricas ni los escenarios monetarios hipotéticos constituyen ganancias realizadas o incrementos causales por el uso de IA.

**Archivos:** [comparación F1 homogénea](resultados/analisis_adicional/comparacion_f1_umbral_validado.csv), [ranking comercial top-20%](resultados/analisis_adicional/metricas_presupuesto20.csv), [análisis por edad](resultados/analisis_adicional/evaluacion_grupos_edad.csv), [sensibilidad económica](resultados/analisis_adicional/sensibilidad_economica_hipotetica.csv), [figura de lift](resultados/figuras/07_lift_presupuesto20.png) y [anexo técnico PDF](docs/Anexo_Mejora_Modelos_Actividad6.pdf). El reporte principal sigue limitado a 7–8 páginas; el anexo es complementario.

## 7. Entregables y formato acordado

- Reporte técnico PDF de 6–8 páginas con interpretación contextual, metodología, métricas, significancia y limitaciones.
- Notebook o script ejecutado y reproducible con CV, ajuste de umbral, comparación y simulación A/B.
- Tablero Trello con criterios, visualizaciones, evidencia de resultados y justificación del modelo. [Enlace al tablero](https://trello.com/b/EIKcvZ7j/banking-conversion-ai-actividad-6).

**Formato de entrega confirmado por el estudiante: únicamente la URL del repositorio GitHub, en sustitución del ZIP.** Los archivos obligatorios permanecen publicados dentro del repositorio. El tablero Trello permanece privado. Las capturas auténticas se incorporarán como evidencias públicas en `evidencias/trello/` y al PDF de ocho páginas una vez recibidas.

## 8. Declaración de uso de inteligencia artificial

| Herramienta | Uso | Verificación |
|---|---|---|
| ChatGPT (OpenAI) | Diseño experimental, scripts, automatización GitHub Actions, organización del tablero Trello, revisión de resultados y preparación del reporte | Todas las cifras proceden de ejecución de código; el estudiante valida conclusiones y autoriza la entrega final. |

## 9. Bibliografía
- Moro, S., Rita, P. y Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5K306
- Documentación de scikit-learn: https://scikit-learn.org/stable/
- Trello: https://trello.com/b/EIKcvZ7j/banking-conversion-ai-actividad-6
