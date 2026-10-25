"""Genera anexo exploratorio del rendimiento, sin reemplazar el reporte central."""
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
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle
)
from .datos import ROOT, OUT

DATA=OUT/"analisis_adicional"
FILE=ROOT/"docs/Anexo_Mejora_Modelos_Actividad6.pdf"

def render():
    required=[
      DATA/"comparacion_f1_umbral_validado.csv",
      DATA/"metricas_presupuesto20.csv",
      DATA/"tercer_modelo_metrics.csv",
      DATA/"baseline_always_positive.json",
      DATA/"evaluacion_grupos_edad.csv",
      OUT/"figuras/07_lift_presupuesto20.png"
    ]
    for x in required:
        if not x.exists():raise FileNotFoundError(str(x))
    f=pd.read_csv(required[0]);top=pd.read_csv(required[1])
    hgb=pd.read_csv(required[2]).iloc[0]
    ref=json.loads(required[3].read_text(encoding="utf-8"))
    age=pd.read_csv(required[4])
    main=pd.read_csv(OUT/"comparacion_modelos.csv").set_index("Modelo")
    assert len(f)==3 and len(top)==3 and ref["f1"]>0
    FILE.parent.mkdir(exist_ok=True,parents=True)
    st=getSampleStyleSheet()
    st.add(ParagraphStyle(name="A6title",fontName="Helvetica-Bold",
        fontSize=19,leading=23,textColor=colors.HexColor("#124558"),spaceAfter=13))
    st.add(ParagraphStyle(name="A6head",fontName="Helvetica-Bold",
        fontSize=11,leading=15,textColor=colors.HexColor("#006B7F"),spaceBefore=8,spaceAfter=7))
    st.add(ParagraphStyle(name="A6body",fontName="Helvetica",
        fontSize=9.15,leading=13.3,spaceAfter=8))
    st.add(ParagraphStyle(name="A6small",fontName="Helvetica",
        fontSize=8.2,leading=11))
    st.add(ParagraphStyle(name="A6caption",fontName="Helvetica-Oblique",
        fontSize=8.2,leading=11,textColor=colors.HexColor("#526A73"),spaceAfter=6))
    doc=SimpleDocTemplate(str(FILE),pagesize=A4,
        topMargin=1.8*cm,bottomMargin=1.7*cm,
        leftMargin=1.95*cm,rightMargin=1.95*cm,
        title="Anexo de evaluación exploratoria de modelos - Actividad 6")
    story=[]
    P=lambda s:Paragraph(s,st["A6body"])
    def para(s):story.append(P(s))
    def head(s):story.append(Paragraph(s,st["A6head"]))
    def tab(headers,rows,widths):
        data=[[Paragraph(html.escape(str(x)),st["A6small"]) for x in headers]]
        data += [[Paragraph(html.escape(str(x)),st["A6small"]) for x in row] for row in rows]
        t=Table(data,colWidths=widths,hAlign="LEFT",repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E5F1F4")),
            ("GRID",(0,0),(-1,-1),.4,colors.HexColor("#C4D9DF")),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("TOPPADDING",(0,0),(-1,-1),6),
            ("BOTTOMPADDING",(0,0),(-1,-1),6)]))
        story.append(t);story.append(Spacer(1,.15*cm))
    def decor(c,d):
        c.saveState()
        w,h=A4
        c.setStrokeColor(colors.HexColor("#006B7F"))
        c.line(2*cm,h-1.3*cm,w-2*cm,h-1.3*cm)
        c.setFont("Helvetica",8)
        c.drawString(2*cm,1.1*cm,"Banking Conversion AI | Anexo exploratorio")
        c.drawRightString(w-2*cm,1.1*cm,str(d.page))
        c.restoreState()

    story.append(Paragraph("ANEXO - ¿POR QUÉ EL F1 ES MODERADO?",st["A6title"]))
    para("<b>Actividad 6.</b> Esta ampliación conserva intactos los resultados y "
         "la selección principal. El test ya se analizó en la versión anterior. "
         "<b>Los resultados que siguen son exploratorios y NO equivalen a "
         "validación independiente.</b>")
    head("1. Referencia del problema desbalanceado")
    para(f"La prevalencia de conversión es 11.70% en el test. Un modelo que "
         f"predice siempre positivo tendría F1={ref['f1']:.4f}; "
         "el baseline siempre negativo tiene F1=0. El F1 original de "
         f"RandomForest fue {main.loc['RandomForest','test_f1']:.4f}, "
         "optimizado en validación para F2, no para F1.")
    head("2. Mismo criterio F1 para tres algoritmos")
    rows=[]
    for _,r in f.iterrows():
        rows.append([
          r.Modelo,
          f"{r.threshold_validation:.2f}",
          f"{r.validation_f1:.4f}",
          f"{r.test_f1_exploratorio:.4f}",
          f"{r.test_precision_exploratorio:.4f}",
          f"{r.test_recall_exploratorio:.4f}"
        ])
    tab(["Modelo","Umbral","F1 val.","F1 test*","Prec. test*","Recall test*"],
        rows,[4.0*cm,2.0*cm,2.4*cm,2.5*cm,2.6*cm,3.1*cm])
    para("*Test exploratorio: para los tres modelos el umbral se determinó "
         "EXCLUSIVAMENTE sobre validación, maximizando F1 sujeto a "
         "precision >=0.20. Las métricas resultantes no se utilizaron "
         "para reoptimizar hiperparámetros. RandomForest conserva la mayor "
         "F1 en este contraste homogéneo.")
    head("3. Costo de la mayor precisión")
    rf=f.set_index("Modelo").loc["RandomForest"]
    para(f"RandomForest pasa de precisión "
         f"{main.loc['RandomForest','test_precision']:.2%} y recall "
         f"{main.loc['RandomForest','test_recall']:.2%} con umbral F2, "
         f"a precisión {rf.test_precision_exploratorio:.2%} "
         f"y recall {rf.test_recall_exploratorio:.2%} "
         "al seleccionar umbral para F1. Mejora precisión y F1, "
         "pero disminuye la cobertura de contrataciones históricas.")
    head("4. Tercer modelo y complejidad")
    para(f"HistGradientBoosting se ajustó con GridSearchCV 5-fold "
         f"(Average Precision CV {hgb.cv_average_precision:.4f}); "
         f"con umbral F2= {hgb.validation_threshold:.2f}, "
         f"F1 exploratorio {hgb.test_f1:.4f}, "
         f"ROC-AUC {hgb.test_roc_auc:.4f}. "
         "Aunque aporta una alternativa competitiva, no supera el "
         "RandomForest con selección homogénea de F1.")

    story.append(PageBreak())
    story.append(Paragraph("APLICACIÓN COMERCIAL SIN AFIRMAR CAUSALIDAD",st["A6title"]))
    head("5. Métricas de priorización bajo capacidad del 20%")
    top_rows=[]
    for _,r in top.iterrows():
        top_rows.append([r.Modelo,str(int(r.seleccionados)),
            str(int(r.positivos_seleccionados)),
            f"{r.precision_at_20:.2%}",f"{r.lift_at_20:.2f}x"])
    tab(["Modelo","Contactos","Respuestas históricas","Prec.@20%","Lift"],
        top_rows,[4.35*cm,2.35*cm,4.5*cm,3.0*cm,2.4*cm])
    img=OUT/"figuras/07_lift_presupuesto20.png"
    w,h=ImageReader(str(img)).getSize()
    scale=min(16.7*cm/w,8.1*cm/h)
    story.append(KeepTogether([
        Image(str(img),width=w*scale,height=h*scale),
        Paragraph("Figura A1. Precision@20% y lift históricos frente a la prevalencia general; no equivalen a uplift causal.",st["A6caption"])
    ]))
    head("6. Análisis de utilidad y límites")
    para("Con un presupuesto fijo de 1,808 contactos sobre 9,043 observaciones "
         "de prueba, los modelos seleccionan aproximadamente 28-30% de "
         "conversiones históricas entre los más puntuados. Esto informa sobre "
         "ranking comercial, pero los datos no incluyen un contrafactual "
         "observado de no contacto ni demuestran mayor ingreso por usar IA.")
    para("Los escenarios económicos de "
         "<b>resultados/analisis_adicional/sensibilidad_economica_hipotetica.csv</b> "
         "suponen costo unitario por contacto y beneficios arbitrarios. "
         "Son ejercicios de sensibilidad, NO dinero efectivamente ganado. "
         "La auditoría por edad es descriptiva, sin inferencia de fairness "
         "suficiente y sin relevancia para decisiones crediticias.")
    head("7. Recomendación")
    para("Conservar el resultado principal y sus umbrales F2 como el análisis "
         "confirmatorio original. Publicar el contraste F1, el tercer modelo "
         "y Precision@20% como análisis exploratorio adicional. La siguiente "
         "fase rigurosa requeriría un test temporal nuevo, identificadores "
         "de clientes para evitar repetición y prueba A/B prospectiva.")

    doc.build(story,onFirstPage=decor,onLaterPages=decor)
    print("Anexo exploratorio generado:",FILE,FILE.stat().st_size)
    return FILE

if __name__=="__main__":
    render()
