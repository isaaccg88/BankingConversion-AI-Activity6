# Banking Conversion AI — Actividad 6

**Gestión de Proyectos de Inteligencia Artificial · Módulo 2 / Semana 7**  
**Contexto:** financiero — contratación de depósitos a plazo mediante campañas telefónicas.  
**Repositorio:** https://github.com/isaaccg88/BankingConversion-AI-Activity6  
**Tablero auténtico de Trello:** https://trello.com/b/EIKcvZ7j/banking-conversion-ai-actividad-6

> **Estado:** scripts y automatizaciones publicados; los resultados se incorporan únicamente después de ejecutar los experimentos y pasar verificaciones de integridad. No se atribuyen tasas ni ganancias ficticias.

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

## 6. Evidencia estadística y conclusiones

Las tablas de resultados, matrices de confusión, gráficos ROC/PR y la comparación A/B se publican **solo después de ejecutar** los experimentos. No se deben confundir p-valores bajos con mejora causal o utilidad empresarial: la simulación usa etiquetas históricas y no contiene un grupo real de control sin contacto.

Para una prueba de campo, habrá que definir previamente población elegible, unidad de aleatorización (cliente), presupuesto, exclusiones, hipótesis y tamaño de muestra, registrar la campaña prospectivamente y utilizar una variable de respuesta causalmente relevante. También será necesaria auditoría de equidad y tratamiento de datos personales.

## 7. Entregables y formato acordado

- Reporte técnico PDF de 6–8 páginas con interpretación contextual, metodología, métricas, significancia y limitaciones.
- Notebook o script ejecutado y reproducible con CV, ajuste de umbral, comparación y simulación A/B.
- Tablero Trello con criterios, visualizaciones, evidencia de resultados y justificación del modelo. [Enlace al tablero](https://trello.com/b/EIKcvZ7j/banking-conversion-ai-actividad-6).

La entrega propuesta es el enlace de GitHub en lugar del ZIP, **sujeto a confirmar que el profesor acepta el mismo cambio para la Actividad 6**. El enlace de Trello privado requiere acceso del profesor, o bien capturas auténticas incluidas como evidencia.

## 8. Declaración de uso de inteligencia artificial

| Herramienta | Uso | Verificación |
|---|---|---|
| ChatGPT (OpenAI) | Diseño experimental, scripts, automatización GitHub Actions, organización del tablero Trello, revisión de resultados y preparación del reporte | Todas las cifras proceden de ejecución de código; el estudiante valida conclusiones y autoriza la entrega final. |

## 9. Bibliografía
- Moro, S., Rita, P. y Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5K306
- Documentación de scikit-learn: https://scikit-learn.org/stable/
- Trello: https://trello.com/b/EIKcvZ7j/banking-conversion-ai-actividad-6
