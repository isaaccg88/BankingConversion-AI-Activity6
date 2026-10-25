# Protocolo experimental preespecificado — Actividad 6

## Pregunta de negocio
¿Puede una política de clasificación de clientes optimizar la selección de llamadas, bajo una capacidad operativa fija, en un conjunto histórico de campañas bancarias? El objetivo NO es demostrar causalidad.

## Hipótesis
H0 predictiva: los errores de RF y LR tienen similar probabilidad de discordancia en test (McNemar). H1: diferente. Se reporta p exacto bilateral y tamaños de efecto (diferencia F1 bootstrap pareado).
H0 descriptiva A/B offline: la tasa histórica en las observaciones seleccionadas por cada estrategia es igual. Se presenta prueba exploratoria y IC95, **no causal**.

## Preprocesamiento
Dataset UCI `bank-full.csv` de 45,211 registros. La variable duration es fuga informativa para predicción antes de llamada; también se excluyen contact/day/month/campaign de un escenario conservador. Las 11 características restantes incluyen historia de contacto previo (pdays/previous/poutcome); no implican inferencia en clientes nunca contactados fuera de distribución.

## Diseño
- Semilla 42, partición estratificada 60/20/20, IDs de fila congelados; train/val/test excluyentes.
- OneHotEncoder/StandardScaler/SimpleImputer dentro de Pipeline. 5-fold CV en train con GridSearchCV y scoring average_precision.
- LogisticRegression variando C y class_weight. RandomForest variando max_depth y min_samples_leaf.
- Umbral sobre validation max F2 sujeto a precision >= 0.20; selección RF vs LR por F2 validación, desempate CV AP.
- Test abierto exactamente una vez para informe, no ajuste.
- Baseline DummyClassifier(strategy='prior').
- McNemar exacto y bootstrap pareado 1,000 iteraciones sobre predicciones del test.
- Simulación A/B: dos subconjuntos test disjuntos, aleatorizados, A selecciona 20% al azar y B selecciona top-20% por score RF; métrica tasa histórica entre seleccionados y diferencias de dos proporciones. No interpretar como resultado de campaña experimental.
- Se reporta limitación grave: solo observamos resultados de contactos históricos; imposible inferir que otro contacto o no contacto habría dado el mismo resultado, además no hay ID cliente para evitar duplicados entre grupos.

## Criterios de rigor académico
Reportar matrices TP/FP/TN/FN, accuracy, precision, recall, F1/F2, balanced accuracy, ROC-AUC, AP, curvas ROC/PR, estimaciones de incertidumbre, posibles falsos positivos como recursos gastados, y falsos negativos como oportunidades omitidas. Documentar sensibilidad del umbral y restricciones operativas. El valor del negocio se expresa en unidades de contactos y tasas históricas; costos monetarios serían supuestos, nunca ingresos probados.

## Experimento A/B auténtico futuro
Unidad aleatoria por cliente (ID único), asignación control A (política actual) vs intervención B (priorización) predefinidas y sin contaminación. Mismo presupuesto y periodo, resultados prospectivos, consentimiento/legalidad, exclusiones, power/tamaño muestral calculados antes, reglas de paro y preregistro. Estimar efecto ITT y ajustar inferencia por multiplicidad si múltiples métricas primarias.

## Trazabilidad
GitHub Actions instala versiones fijas, publica CSV de 45,211 filas SHA256, splits, métricas, predicciones, figuras y notebook auditor con salidas reales. Controlar documentación del reporte y capturas o enlace accesible al tablero Trello.

## Atribución
Moro et al., Bank Marketing, UCI Machine Learning Repository, DOI 10.24432/C5K306, CC BY 4.0.
