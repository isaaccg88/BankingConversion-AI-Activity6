## 2026-10-25 — Auditoría adicional del F1 y ranking comercial
- Se evaluó HistGradientBoosting y se recalculó para los tres modelos el F1 con **el mismo objetivo de selección de umbral en validación**, sin modificar el entrenamiento o las conclusiones originales.
- F1 test exploratorio: LR 0.3662; RF 0.4117; HGB 0.3990. Se explicita que el test ya se inspeccionó previamente y no constituye validación independiente.
- Se añadieron baseline siempre positivo (F1 0.2095), Precision@20%, lift, segmentación descriptiva por edad y escenarios económicos hipotéticos sin inferencia causal.
- [Anexo PDF de dos páginas](docs/Anexo_Mejora_Modelos_Actividad6.pdf) generado y renderizado; su auditoría de 2 páginas e imagen integrada pasó GitHub Actions.
- Se habilitó la incorporación automática de dos **capturas reales** del tablero Trello como octava página del PDF principal al subirlas a `evidencias/trello/`.
- Se creó una tarjeta adicional Trello con resultados exploratorios y diferenciación explícita entre clasificación y uplift causal.
- El formato final de entrega es la URL del repositorio de GitHub y no un ZIP, conforme a lo indicado por el estudiante.

# Registro de cambios — Banking Conversion AI / Actividad 6

## 2026-10-25 — Versión de evaluación y validación (resultados reales)
- Se seleccionó **Bank Marketing UCI 222** para cumplir el ámbito financiero exigido. Se corrigió el DOI oficial a **10.24432/C5K306**, licencia CC BY 4.0.
- Se conservaron los 45,211 registros (`bank-full.csv`, 17 columnas) con SHA256. Se utilizan once predictores conservadores disponibles antes del contacto; `duration` y características de la campaña actual se excluyen para prevenir fuga.
- Se congelaron particiones estratificadas de train 27,126, validación 9,042 y test 9,043, semilla 42.
- Se entrenaron LogisticRegression y RandomForest con GridSearchCV de cinco folds, y DummyClassifier como baseline.
- Umbrales seleccionados únicamente en validación: LR=0.11, RF=0.12. Se eligió RandomForest por F2 de validación (0.4809 frente a 0.4640).
- En test: RF F1=0.3545, ROC-AUC=0.7392; LR F1=0.3124, ROC-AUC=0.7204. LR ofrece mayor recall: 0.6474 vs RF 0.5756, limitación explícita en la conclusión.
- Se ejecutaron McNemar exacto y bootstrap pareado de diferencia F1, y una simulación A/B **solo offline y no causal** con política A aleatoria vs ranking RF.
- GitHub Actions verificó prueba de código, integridad del dataset, notebook de ocho celdas ejecutadas sin errores, PDF de siete páginas y cinco imágenes.
- Se creó el tablero privado Trello Banking Conversion AI — Actividad 6 y se cargaron inicialmente diez tarjetas; se añadió una undécima con el análisis exploratorio del F1 y Precision@20%.
- Se publicó reporte [PDF](docs/Reporte_Tecnico_Actividad6.pdf) y notebook [ejecutado](notebooks/Evaluacion_Actividad6.ipynb).

## Pendientes de entrega administrativa
- Asegurar que el docente puede **abrir Trello privado**, o adjuntar capturas genuinas en el repositorio, dado que las instrucciones admiten capturas en PDF o enlace.
- Se confirma entrega mediante enlace al repositorio GitHub, sin ZIP.
- Realizar revisión personal final del PDF, hipótesis y conclusiones, antes de enviarlo.

## Declaración de asistencia por IA
ChatGPT (OpenAI): asistencia en investigación, diseño, programación, documentación, organización en Trello y auditoría. Todas las cifras empíricas proceden de la ejecución automatizada registrada en GitHub Actions; la interpretación y entrega corresponden al estudiante.
