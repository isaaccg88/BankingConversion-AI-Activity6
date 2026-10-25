"""Reporte académico de 7 páginas basado EXCLUSIVAMENTE en métricas reales.

python -m fuentes.generar_reporte
La simulación A/B offline no permite inferencias causales.
"""
from __future__ import annotations
import html
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle
)
from .datos import ROOT, OUT

PDF=ROOT/"docs/Reporte_Tecnico_Actividad6.pdf"
TEAL=colors.HexColor("#006B7F")
INK=colors.HexColor("#183744")
GRAY=colors.HexColor("#5B6D75")
LIGHT=colors.HexColor("#EAF3F5")
FIG=OUT/"figuras"

def esc(x): return html.escape(str(x))
def make():
    required=["comparacion_modelos.csv","baseline_dummy.json","fuente_y_calidad.json",
              "simulacion_ab.json","mcnemar.json","bootstrap_pareado.csv",
              "seleccion_congelada.json","resumen.json"]
    missing=[n for n in required if not (OUT/n).is_file()]
    if missing: raise FileNotFoundError("Faltan experimentos reales: "+str(missing))
    df=pd.read_csv(OUT/"comparacion_modelos.csv").set_index("Modelo")
    lr,rf=df.loc["LogisticRegression"],df.loc["RandomForest"]
    source=json.loads((OUT/"fuente_y_calidad.json").read_text(encoding="utf8"))
    b=json.loads((OUT/"baseline_dummy.json").read_text(encoding="utf8"))
    m=json.loads((OUT/"mcnemar.json").read_text(encoding="utf8"))
    ab=json.loads((OUT/"simulacion_ab.json").read_text(encoding="utf8"))
    boot=pd.read_csv(OUT/"bootstrap_pareado.csv").iloc[0]
    chosen=json.loads((OUT/"seleccion_congelada.json").read_text(encoding="utf8"))
    assert len(df)==2 and ab["tipo"]=="simulacion_AB_offline_no_causal"
    PDF.parent.mkdir(parents=True,exist_ok=True)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BCATitle",fontName="Helvetica-Bold",fontSize=23,
            leading=28,alignment=TA_CENTER,textColor=INK,spaceAfter=14))
    styles.add(ParagraphStyle(name="BCAHead",fontName="Helvetica-Bold",fontSize=12.5,
            leading=16,textColor=TEAL,spaceBefore=10,spaceAfter=8,keepWithNext=True))
    styles.add(ParagraphStyle(name="BCAPara",fontName="Helvetica",fontSize=9.25,
            leading=13.3,textColor=INK,spaceAfter=8))
    styles.add(ParagraphStyle(name="BCASmall",fontName="Helvetica",fontSize=8,
            leading=11,textColor=GRAY,spaceAfter=5))
    styles.add(ParagraphStyle(name="BCACaption",fontName="Helvetica-Oblique",fontSize=8,
            leading=11,textColor=GRAY,spaceAfter=9))
    doc=SimpleDocTemplate(str(PDF),pagesize=A4,
        topMargin=1.75*cm,bottomMargin=1.75*cm,
        leftMargin=2*cm,rightMargin=2*cm,
        title="Banking Conversion AI - Evaluación y validación - Actividad 6",
        author="Proyecto académico Banking Conversion AI")
    story=[]
    P=lambda x:Paragraph(x,styles["BCAPara"])
    def para(s):story.append(P(s))
    def head(s):story.append(Paragraph(s,styles["BCAHead"]))
    def caption(s):story.append(Paragraph(s,styles["BCACaption"]))
    def new():story.append(PageBreak())
    def tab(headers, rows, widths=None):
        values=[[Paragraph(esc(s),styles["BCASmall"]) for s in headers]]
        values += [[Paragraph(esc(s),styles["BCASmall"]) for s in row]
                   for row in rows]
        t=Table(values,colWidths=widths,repeatRows=1,hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),LIGHT),
            ("GRID",(0,0),(-1,-1),.4,colors.HexColor("#CCD9DF")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LEFTPADDING",(0,0),(-1,-1),7),
            ("TOPPADDING",(0,0),(-1,-1),6),
            ("BOTTOMPADDING",(0,0),(-1,-1),5)]))
        story.append(t)
        story.append(Spacer(1,.17*cm))
    def figure(name,title,maxwidth=16.7*cm,maxheight=9*cm):
        path=FIG/name
        if not path.exists():raise FileNotFoundError(path)
        px,py=ImageReader(str(path)).getSize()
        scale=min(maxwidth/px,maxheight/py)
        pic=Image(str(path),width=px*scale,height=py*scale)
        story.append(KeepTogether([pic,Paragraph(title,styles["BCACaption"])]))

    # Page 1
    story.append(Spacer(1,2.15*cm))
    story.append(Paragraph("BANKING CONVERSION AI",styles["BCATitle"]))
    story.append(Paragraph("Actividad 6 — Evaluación y validación estadística de clasificadores",styles["BCAHead"]))
    para("<b>Gestión de Proyectos de Inteligencia Artificial · Módulo 2 / Semana 7.</b>")
    para("Contexto: institución bancaria portuguesa, contratación de depósitos a plazo. "
         "Fuente: UCI Bank Marketing (45,211 registros), licencia CC BY 4.0, "
         "DOI 10.24432/C5K306. Objetivo SMART y protocolo completos en el README.")
    head("Resumen ejecutivo")
    para("Se evaluaron regresión logística y Random Forest frente a un baseline "
         "con particiones separadas, validación cruzada estratificada de cinco folds "
         "y ajuste de umbral exclusivamente con validación. El objetivo de negocio "
         "es priorizar contactos comerciales bajo un presupuesto de llamadas, "
         "sin afirmar efecto causal sobre contrataciones.")
    tab(["Indicador test","Regresión logística","Random Forest"],[
        ["F1",f"{lr.test_f1:.4f}",f"{rf.test_f1:.4f}"],
        ["Recall",f"{lr.test_recall:.4f}",f"{rf.test_recall:.4f}"],
        ["ROC-AUC",f"{lr.test_roc_auc:.4f}",f"{rf.test_roc_auc:.4f}"],
        ["Average Precision",f"{lr.test_average_precision:.4f}",f"{rf.test_average_precision:.4f}"]
    ],[6.0*cm,5.2*cm,5.2*cm])
    para(f"Se preseleccionó <b>{esc(chosen['modelo_seleccionado_en_validacion'])}</b> "
         f"con umbral <b>{chosen['umbral']:.2f}</b>. El test no se utilizó para elegir ese umbral. "
         "Las tasas A/B presentadas son HISTÓRICAS y proceden de simulación offline, "
         "no de una campaña aleatorizada prospectiva.")
    new()
    # Page 2
    head("1. Problema, objetivo SMART y datos")
    para("Problema: la capacidad de llamadas es limitada. Se busca comparar modelos "
         "para ordenar contactos antes de realizar la llamada. Objetivo específico: "
         "evaluar tres algoritmos, matriz de confusión, ROC/AUC, umbrales y validación "
         "cruzada sobre Bank Marketing, con test reservado y una simulación A/B del "
         "20% de capacidad; medir las diferencias y documentar límites de inferencia.")
    para(f"El conjunto contiene <b>{source['filas']:,}</b> observaciones, "
         f"<b>{source['positivos']:,}</b> conversiones históricas "
         f"({source['positivos']/source['filas']:.2%}) y "
         f"{source['negativos']:,} no conversiones. "
         "Se preservan el archivo original y SHA256, con 11 predictoras "
         "operativas plausibles antes del contacto.")
    tab(["Etapa","Registros","Objetivo"],[
        ["Train","27,126","Grid Search y CV estratificada"],
        ["Validation","9,042","Selección de umbral y modelo"],
        ["Test","9,043","Medición final, McNemar, bootstrap y A/B offline"]
    ],[3.8*cm,3.5*cm,9.1*cm])
    head("Control de fuga y transformación")
    para("Se excluye <b>duration</b> porque no puede conocerse antes de una llamada; "
         "también contact/day/month/campaign ante ambigüedad de disponibilidad previa. "
         "Las 11 variables restantes incluyen historia bancaria y de campañas pasadas. "
         "Imputación, escalado y OneHotEncoder se ajustan dentro de Pipeline/CV. "
         "No existe identificador de cliente, por lo que no podemos excluir "
         "con certeza que una persona se repita entre particiones.")
    para("Otra limitación: los registros históricos del archivo antiguo son observacionales "
         "y su distribución puede diferir de campañas futuras. "
         "La prueba aleatoria no sustituye una partición por fecha ni datos prospectivos.")
    head("Criterios predefinidos")
    para("GridSearchCV maximiza Average Precision. El umbral operativo se selecciona "
         "en validación maximizando F2 sujeto a precision al menos 0.20, si es posible. "
         "La elección entre modelos maximiza F2 en validación, con desempate por AP de CV.")
    new()
    # Page 3
    head("2. Matrices de confusión y métricas")
    tab(["Métrica test","Logística","Random Forest"],[
       ["Accuracy",f"{lr.test_accuracy:.3f}",f"{rf.test_accuracy:.3f}"],
       ["Precision",f"{lr.test_precision:.3f}",f"{rf.test_precision:.3f}"],
       ["Recall",f"{lr.test_recall:.3f}",f"{rf.test_recall:.3f}"],
       ["F1 / F2",f"{lr.test_f1:.3f} / {lr.test_f2:.3f}",
                   f"{rf.test_f1:.3f} / {rf.test_f2:.3f}"],
       ["Especificidad",f"{lr.test_specificity:.3f}",f"{rf.test_specificity:.3f}"],
       ["TN / FP / FN / TP",f"{int(lr.test_tn)} / {int(lr.test_fp)} / {int(lr.test_fn)} / {int(lr.test_tp)}",
                           f"{int(rf.test_tn)} / {int(rf.test_fp)} / {int(rf.test_fn)} / {int(rf.test_tp)}"]
    ],[5.2*cm,5.5*cm,5.7*cm])
    figure("03_matrices_confusion.png",
           "Figura 3. Matrices de confusión calculadas sobre test a los umbrales congelados.",maxheight=8.1*cm)
    para(f"Interpretación financiera: Random Forest detectó <b>{int(rf.test_tp)}</b> "
         f"conversiones históricas y omitió {int(rf.test_fn)}; la logística "
         f"detectó {int(lr.test_tp)} y omitió {int(lr.test_fn)}. "
         f"Random Forest generó {int(rf.test_fp)} contactos priorizados sin conversión histórica "
         f"frente a {int(lr.test_fp)} de la logística. "
         "Un falso positivo representa costo potencial de contacto, no fraude ni daño clínico. "
         "El umbral cambia la relación entre cobertura y eficiencia.")
    new()
    # Page 4
    head("3. Curvas ROC/AUC y Precision-Recall")
    para("ROC-AUC mide capacidad discriminativa entre positivos y negativos a "
         "todos los umbrales, mientras Average Precision resume precisión-recall "
         "en una clase minoritaria (conversiones históricas). El modelo se comparó "
         "sobre los mismos 9,043 registros de test.")
    figure("01_02_curvas_roc_pr.png",
           "Figuras 1 y 2. ROC y Precision-Recall obtenidas de probabilidades reales de prueba.",maxheight=8.5*cm)
    para(f"Random Forest logró ROC-AUC <b>{rf.test_roc_auc:.4f}</b> "
         f"frente a {lr.test_roc_auc:.4f} de la logística, "
         f"y AP <b>{rf.test_average_precision:.4f}</b> frente a "
         f"{lr.test_average_precision:.4f}. "
         "La ventaja en ranking no equivale por sí sola a un mayor número "
         "de contrataciones atribuibles a utilizar IA.")
    head("4. Umbral de decisión")
    para(f"Los umbrales de la evaluación fueron <b>{lr.validation_threshold:.2f}</b> "
         f"(logística) y <b>{rf.validation_threshold:.2f}</b> (Random Forest). "
         f"Se eligieron en validación, donde F2 fue {lr.validation_f2:.4f} "
         f"y {rf.validation_f2:.4f}, respectivamente. "
         "Los umbrales relativamente bajos permiten captar más positivos "
         "a costa de falsas alarmas. El umbral no fue reoptimizado con test.")
    new()
    # Page 5
    head("5. Validación cruzada, baseline y significancia")
    para("Se entrenaron dos familias de algoritmos usando GridSearchCV "
         "y StratifiedKFold(5), con preprocesamiento sin fuga dentro del pipeline. "
         "Se seleccionaron parámetros maximizando Average Precision en train; "
         "la elección final quedó separada de test.")
    tab(["Modelo","AP media CV","F2 validación","Tiempo ajuste (s)"],[
        ["Logística",f"{lr.cv_average_precision:.4f}",f"{lr.validation_f2:.4f}",f"{lr.fit_seconds:.1f}"],
        ["Random Forest",f"{rf.cv_average_precision:.4f}",f"{rf.validation_f2:.4f}",f"{rf.fit_seconds:.1f}"],
    ],[4.2*cm,3.9*cm,4*cm,4.3*cm])
    figure("04_cv.png","Figura 4. Average Precision de configuraciones en validación cruzada.",maxheight=5.6*cm)
    para(f"Baseline DummyClassifier: accuracy {b['accuracy']:.3f}, "
         f"recall {b['recall']:.3f} y F1 {b['f1']:.3f}. "
         "Una exactitud alta sin detección de conversiones es insuficiente.")
    para(f"Prueba exacta de McNemar sobre errores discordantes: "
         f"{m['a_correcto_b_error']} casos correctos solo para logística y "
         f"{m['a_error_b_correcto']} correctos solo para Random Forest; "
         f"p={m['p_exacto_bilateral']:.2g}. "
         "McNemar contrasta diferencias en errores, no en dinero ganado.")
    para(f"Bootstrap pareado (n={int(boot['repeticiones'])}): "
         f"diferencia F1(RF - logística) = {boot['delta_f1']:.4f}, "
         f"IC95 percentil [{boot['ic95_inferior']:.4f}, {boot['ic95_superior']:.4f}]. "
         "La incertidumbre estadística no subsana sesgo de muestreo o drift.")
    new()
    # Page 6
    head("6. Simulación A/B offline — no causal")
    para("Se dividió aleatoriamente el test en dos grupos disjuntos. A seleccionó "
         "20% de contactos al azar; B seleccionó 20% según el score de Random Forest. "
         "Las etiquetas corresponden a contrataciones históricas observadas, "
         "no al resultado de asignar realmente esas políticas a clientes.")
    tab(["Medida","Política A","Política B"],[
        ["Observaciones del grupo",str(ab["n_grupo_A"]),str(ab["n_grupo_B"])],
        ["Contactos seleccionados",str(ab["contactados_A"]),str(ab["contactados_B"])],
        ["Contrataciones históricas",str(ab["conversiones_historicas_A"]),str(ab["conversiones_historicas_B"])],
        ["Tasa en seleccionados",f"{ab['tasa_A']:.2%}",f"{ab['tasa_B']:.2%}"]
    ],[5.8*cm,5.1*cm,5.5*cm])
    figure("05_ab_offline.png",
           "Figura 5. Tasa de respuesta histórica entre seleccionados; NO representa uplift causal.",maxheight=6.8*cm)
    para(f"Diferencia descriptiva de tasas: {ab['diferencia_tasas']:.2%} "
         f"(IC95 aproximado {ab['ic95_inf']:.2%} a {ab['ic95_sup']:.2%}); "
         f"p exploratorio={ab['p_z_exploratorio']:.2g}. "
         "El contraste de proporciones entre observaciones seleccionadas "
         "no estima causalidad: existe sesgo de selección condicionado al score "
         "y no hay contrafactual de no llamar.")
    head("Diseño para una prueba real")
    para("Aleatorizar por cliente único, definir campaña y período prospectivos, "
         "misma capacidad por grupo, métrica primaria pre-registrada, control de "
         "interferencia, consentimiento y verificación legal, tamaño de muestra "
         "basado en potencia y reglas de paro. Comparar por intención de tratar "
         "con auditoría de equidad y monitoreo temporal.")
    new()
    # Page 7
    head("7. Recomendación de negocio y límites")
    para(f"Modelo preseleccionado en validación: <b>{esc(chosen['modelo_seleccionado_en_validacion'])}</b> "
         f"(F2 validación {rf.validation_f2:.4f}). "
         "Logística obtiene mayor recall en test, "
         "por lo que una institución más adversa a perder clientes potenciales "
         "podría preferirla, pese a su menor precisión y F1. "
         "La decisión debe incorporar presupuesto, costo por llamada y valoración "
         "de oportunidades perdidas, aún no cuantificados con datos reales.")
    figure("06_f1_test.png","Figura 6. Diferencia observada en F1 sobre conjunto de prueba.",maxheight=5.6*cm)
    head("8. Evidencias, deuda técnica y Trello")
    para("Trazabilidad: código, notebook ejecutado, CSV original, splits, matrices, "
         "curvas, hipótesis y exportaciones reproducibles en GitHub. "
         "Tablero Trello: <link href='https://trello.com/b/EIKcvZ7j/banking-conversion-ai-actividad-6'>"
         "Banking Conversion AI — Actividad 6</link> (el profesor requiere acceso "
         "a un tablero privado o capturas genuinas adjuntas).")
    para("Limitaciones: datos observacionales antiguos, cambios de campaña, "
         "ausencia de ID de cliente para deduplicación, variables sensibles o "
         "correlacionadas con características demográficas, falta de calibración "
         "económica y ausencia de validación prospectiva. "
         "Mejoras MLOps: monitorizar drift, calibración, precisión por segmento, "
         "trazabilidad de datos, umbral y equidad; conservar supervisión humana.")
    head("Declaración de uso de IA y referencias")
    para("<b>ChatGPT:</b> asistencia en diseño experimental, código, documentación, "
         "tablero Trello y auditoría de resultados. Las métricas proceden de "
         "ejecución reproducible, sin valores generados manualmente.")
    story.append(Paragraph(
         "Moro, S., Rita, P. y Cortez, P. (2014). Bank Marketing [Dataset]. "
         "UCI Machine Learning Repository. https://doi.org/10.24432/C5K306",
         styles["BCASmall"]))
    story.append(Paragraph("Código y evidencia: https://github.com/isaaccg88/BankingConversion-AI-Activity6",
                           styles["BCASmall"]))

    def decor(c,d):
        c.saveState()
        w,h=A4
        c.setStrokeColor(TEAL);c.setLineWidth(.8)
        c.line(2*cm,h-1.35*cm,w-2*cm,h-1.35*cm)
        c.setFont("Helvetica",8);c.setFillColor(GRAY)
        c.drawString(2*cm,1.1*cm,"Banking Conversion AI · Actividad 6")
        c.drawRightString(w-2*cm,1.1*cm,f"Página {d.page}")
        c.restoreState()
    doc.build(story,onFirstPage=decor,onLaterPages=decor)
    print("PDF generado desde resultados medidos:",PDF,PDF.stat().st_size,"bytes")
    return PDF

if __name__=="__main__":
    make()
