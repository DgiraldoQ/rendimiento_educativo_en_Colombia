import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==================== CARGA DE DATOS CORREGIDA ====================
def load_and_clean_data():
    """Carga y limpia EDUCACION.csv con punto y coma como delimitador"""
    try:
        # Cargar CSV con punto y coma como delimitador
        df = pd.read_csv('EDUCACION.csv', encoding='latin-1', sep=';', on_bad_lines='skip')
        
        print(f"📊 Columnas encontradas: {len(df.columns)}")
        print(f"📋 Primeras columnas: {df.columns.tolist()[:5]}")
        
        # Función para limpiar porcentajes: "86,93%" → 0.8693
        def clean_percentage(value):
            if pd.isna(value) or value == '' or str(value).strip() == '':
                return np.nan
            try:
                # Limpiar: quitar %, cambiar coma por punto, convertir a float
                clean = str(value).replace('%', '').replace(',', '.').strip()
                num = float(clean)
                # Si el valor parece estar en porcentaje (ej: 86.93), convertir a proporción
                if num > 1:
                    return num / 100
                return num
            except:
                return np.nan
        
        # Columnas que son porcentajes
        percentage_cols = [col for col in df.columns if any(kw in str(col) for kw in 
                           ['COBERTURA', 'DESERCIÓN', 'APROBACIÓN', 'REPROBACIÓN', 'REPITENCIA', 'MATRICULACIÓN'])]
        
        for col in percentage_cols:
            if col in df.columns:
                df[col] = df[col].apply(clean_percentage)
        
        # Columnas numéricas directas
        numeric_cols = ['AÑO', 'CÓDIGO_MUNICIPIO', 'CÓDIGO_DEPARTAMENTO', 'POBLACIÓN_5_16', 'CÓDIGO_ETC']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filtrar filas válidas
        required = ['APROBACIÓN_MEDIA', 'DESERCIÓN_MEDIA', 'REPROBACIÓN', 'DEPARTAMENTO']
        for col in required:
            if col not in df.columns:
                print(f"❌ Columna faltante: {col}")
                return None
        
        df = df.dropna(subset=required)
        
        # Mantener valores entre 0 y 1
        for col in ['APROBACIÓN_MEDIA', 'DESERCIÓN_MEDIA', 'REPROBACIÓN']:
            df = df[(df[col] >= 0) & (df[col] <= 1)]
        
        print(f"✅ Datos cargados: {len(df)} filas válidas")
        print(f"📊 Departamentos únicos: {df['DEPARTAMENTO'].nunique()}")
        return df
        
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        import traceback
        traceback.print_exc()
        return None

df_global = load_and_clean_data()

# ==================== CARGA DEL MODELO ====================
def load_model():
    try:
        model = joblib.load('model_boxcox_reduced_robust.pkl')
        print("✅ Modelo cargado")
        return model
    except Exception as e:
        print(f"⚠️ Error cargando modelo: {e}")
        return None

model_global = load_model()

# ==================== TAB 1: VISUALIZACIONES ====================
def create_visualizations():
    """Carga datos y genera gráficos"""
    df = df_global  # Usar variable global cargada al inicio
    
    if df is None:
        return "❌ Error cargando datos. Verifica EDUCACION.csv", None, None, None, None, None
    
    try:
        # KPIs
        kpi_text = f"""
### 📊 Indicadores Nacionales Educativos

| Indicador | Valor Nacional |
|-----------|---------------|
| 📈 **Aprobación Media** | {df['APROBACIÓN_MEDIA'].mean()*100:.1f}% |
| 📉 **Deserción Media** | {df['DESERCIÓN_MEDIA'].mean()*100:.1f}% |
| ⚠️ **Reprobación** | {df['REPROBACIÓN'].mean()*100:.1f}% |
| 🏫 **Municipios** | {df['MUNICIPIO'].nunique()} |
| 🗺️ **Departamentos** | {df['DEPARTAMENTO'].nunique()} |
"""
        
        # 1. Top Departamentos
        dept_approval = df.groupby('DEPARTAMENTO')['APROBACIÓN_MEDIA'].agg(['mean', 'count']).reset_index()
        dept_approval = dept_approval[dept_approval['count'] >= 3]
        top_depts = dept_approval.nlargest(10, 'mean')
        
        fig_bar = px.bar(
            x=top_depts['mean']*100, y=top_depts['DEPARTAMENTO'], orientation='h',
            title='🏆 Top 10 Departamentos con Mayor Aprobación',
            color=top_depts['mean']*100, color_continuous_scale='Blues',
            labels={'x': 'Aprobación (%)', 'y': 'Departamento'}
        )
        fig_bar.update_layout(height=400, template='plotly_white', showlegend=False, xaxis=dict(range=[0,100]))
        
        # 2. Scatter Deserción vs Aprobación
        fig_scatter = px.scatter(
            df, x='DESERCIÓN_MEDIA', y='APROBACIÓN_MEDIA', color='DEPARTAMENTO',
            hover_data=['MUNICIPIO', 'AÑO'], opacity=0.7,
            title='🎯 Deserción vs Aprobación por Municipio',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_scatter.update_layout(template='plotly_white', height=400)
        
        # 3. Boxplot por Departamento
        top_names = top_depts['DEPARTAMENTO'].tolist()
        df_filt = df[df['DEPARTAMENTO'].isin(top_names)]
        fig_box = px.box(
            df_filt, x='DEPARTAMENTO', y='APROBACIÓN_MEDIA', color='DEPARTAMENTO',
            title='📦 Distribución de Aprobación (Top 10 Departamentos)', points='outliers'
        )
        fig_box.update_layout(template='plotly_white', height=400, showlegend=False)
        
        # 4. Heatmap correlaciones
        corr_cols = ['APROBACIÓN_MEDIA', 'DESERCIÓN_MEDIA', 'REPROBACIÓN', 'COBERTURA_NETA']
        corr_cols = [c for c in corr_cols if c in df.columns]
        if len(corr_cols) >= 3:
            fig_heat = px.imshow(df[corr_cols].corr(), text_auto='.2f', color_continuous_scale='RdBu_r',
                                title='🔥 Correlaciones entre Indicadores', zmin=-1, zmax=1)
            fig_heat.update_layout(template='plotly_white', height=350)
        else:
            fig_heat = None
        
        # 5. Línea temporal
        if 'AÑO' in df.columns:
            yearly = df.groupby('AÑO')['APROBACIÓN_MEDIA'].mean().reset_index()
            fig_line = px.line(yearly, x='AÑO', y='APROBACIÓN_MEDIA', markers=True,
                              title='📈 Evolución de Aprobación Media por Año')
            fig_line.update_layout(template='plotly_white', height=350, yaxis=dict(range=[0.5,1]))
        else:
            fig_line = None
        
        return kpi_text, fig_bar, fig_scatter, fig_box, fig_heat, fig_line
        
    except Exception as e:
        print(f"❌ Error en visualizaciones: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", None, None, None, None, None

# ==================== TAB 2: SIMULADOR CORREGIDO ====================
def predict_approval(reprobacion, desercion):
    """Predicción robusta usando modelo PKL + lógica educativa"""

    try:
        import numpy as np
        import pandas as pd
        import joblib

        # ================== CARGAR MODELO ==================
        model = joblib.load('model_boxcox_reduced_robust.pkl')

        # ================== VALIDACIÓN ==================
        if reprobacion is None or desercion is None:
            return "⚠️ Ingresa valores numéricos"

        # ================== BOXCOX ==================
        lambda_box = getattr(model, 'lambda_boxcox', getattr(model, '_lambda', 0.342))
        epsilon = 1e-6

        def boxcox_transform(x, lam):
            x_safe = max(float(x), epsilon)
            if abs(lam) < 1e-10:
                return np.log(x_safe)
            return (x_safe**lam - 1) / lam

        rep_t = boxcox_transform(reprobacion, lambda_box)
        des_t = boxcox_transform(desercion, lambda_box)

        # ================== PREDICCIÓN (CLAVE) ==================
        if hasattr(model, "predict") and hasattr(model, "model"):

            exog_names = model.model.exog_names

            data = {}
            for name in exog_names:
                if name.lower() in ['const', 'intercept']:
                    data[name] = 1
                elif 'REPROB' in name.upper():
                    data[name] = rep_t
                elif 'DESERC' in name.upper():
                    data[name] = des_t
                else:
                    data[name] = 0

            X = pd.DataFrame([data])
            y_log = model.predict(X)[0]
            y_pred = float(np.exp(y_log))

        else:
            return f"⚠️ Modelo no compatible: {type(model)}"

        # ================== AJUSTE REALISTA ==================
        suma = reprobacion + desercion
        limite_teorico = max(0, 1 - suma)

        aprobacion_realista = min(y_pred, limite_teorico)

        # ================== DIAGNÓSTICO ==================
        if aprobacion_realista < 0.6:
            nivel = "🔴 CRÍTICO"
            accion = "Intervención urgente: alto riesgo de fracaso escolar."

        elif aprobacion_realista < 0.75:
            nivel = "🟡 BAJO"
            accion = "Reducir deserción y reforzar aprendizajes básicos."

        elif aprobacion_realista < 0.9:
            nivel = "🟢 ACEPTABLE"
            accion = "Optimizar calidad educativa y seguimiento focalizado."

        else:
            nivel = "🔵 ALTO"
            accion = "Mantener y escalar buenas prácticas."

        # ================== RESULTADO ==================
        return f"""
### 🎯 Resultado

- **Aprobación estimada:** {aprobacion_realista:.2%}

### 🧠 Diagnóstico
- **Nivel:** {nivel}

### 💡 Acción recomendada
{accion}

### 📌 Contexto
- Reprobación: {reprobacion:.2%}
- Deserción: {desercion:.2%}
- Pérdida total: {suma:.2%}
- Máximo posible de aprobación: {limite_teorico:.2%}
"""

    except FileNotFoundError:
        return "⚠️ No se encontró el archivo del modelo (.pkl)"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error en predicción: {str(e)}"

# ==================== TAB 3: HALLAZGOS ====================
def get_findings():
    return """
# 📋 Informe Analítico de la Educación en Colombia

## 🚨 La problemática real

En Colombia, el desafío educativo no es solo el rendimiento académico.  
Es estructural.

👉 Miles de estudiantes no terminan el ciclo educativo.  
👉 Otros permanecen, pero no logran aprobar.  

**Ambos fenómenos afectan directamente la calidad del sistema educativo.**

---

## ❓ Pregunta de Investigación

**¿Qué factor impacta más la aprobación educativa: la deserción o la reprobación?**

---

## 📊 Resultados del Modelo

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **R² Ajustado** | 0.677 | Alta capacidad explicativa |
| **MAE** | 0.015 | Error bajo (±1.5 pp) |
| **Durbin-Watson** | 2.018 | Sin autocorrelación |
| **VIF** | ≈1.0 | Variables independientes |

---

## 🔍 Hallazgos Clave

### 🔴 1. La deserción es el factor dominante
- Impacto: **-1.11 puntos por cada 1%**
- Es el principal determinante de la caída en aprobación

💥 **Interpretación:**  
Si un estudiante abandona, el sistema pierde completamente su posibilidad de éxito.

---

### 🟡 2. La reprobación también importa
- Impacto: **-0.75 puntos por cada 1%**
- Influye directamente en el rendimiento

📌 Pero su efecto es menor comparado con la deserción.

---

### 🟢 3. No todos los municipios son iguales
- Mayor incertidumbre en contextos críticos
- Resultados deben interpretarse con contexto local

---

### 🔵 4. Modelo estadísticamente sólido
- Transformación Box-Cox aplicada correctamente
- Inferencias confiables

---

## 🧠 Insight Estratégico

👉 **No todos los problemas educativos pesan igual.**

Reducir la deserción tiene un impacto más fuerte que mejorar el rendimiento académico.

---

## 💡 Implicaciones de Política Pública

### 🥇 Prioridad 1: Retención escolar
- Incentivos económicos
- Transporte y alimentación
- Programas de permanencia

### 🥈 Prioridad 2: Mejora académica
- Refuerzo educativo
- Formación docente

---

## ⚠️ Advertencias

- Evitar interpretar resultados en extremos sin contexto
- Complementar con análisis cualitativo
- Usar el modelo como guía, no como verdad absoluta

---

## 🎯 Conclusión

El problema no es solo que los estudiantes reprueben.

👉 **El verdadero problema es que abandonan el sistema.**

---

🔗 Metodología completa:  
https://www.kaggle.com/code/dgiraldoq/an-lisis-de-factores-asociados-a-la-educacion-co
"""

# ==================== INTERFAZ PRINCIPAL ====================
with gr.Blocks(title="EduPredict Colombia", theme=gr.themes.Soft(primary_hue="blue"),
               css=".gradio-container { max-width: 1400px !important; } h1 { text-align: center; color: #1E40AF; }") as app:
    
    gr.Markdown("# 🎓 EduPredict Colombia")
    gr.Markdown("Modelo predictivo para política educativa municipal")
    
    with gr.Tabs():
        
        # TAB 1
        with gr.TabItem("📊 Visualización"):
            gr.Markdown("### Exploración Interactiva del Dataset Educativo Nacional")
            kpi_out = gr.Markdown()
            with gr.Row(): bar_plot = gr.Plot(label="🏆 Top Departamentos")
            with gr.Row(): scatter_plot = gr.Plot(label="🎯 Deserción vs Aprobación")
            with gr.Row(): box_plot = gr.Plot(label="📦 Distribución por Departamento")
            with gr.Row():
                heat_plot = gr.Plot(label="🔥 Correlaciones")
                line_plot = gr.Plot(label="📈 Evolución Temporal")
            app.load(fn=create_visualizations, inputs=[], outputs=[kpi_out, bar_plot, scatter_plot, box_plot, heat_plot, line_plot])
        
        # TAB 2
        with gr.TabItem("🔮 Simulador"):
            gr.Markdown("### Simulador de Escenarios Educativos\n\n**TASA DE APROBACIÓN_MEDIA:**")
            with gr.Row():
                inp_rep = gr.Number(label="Reprobación", value=0.05)
                inp_des = gr.Number(label="Deserción", value=0.08)
                btn = gr.Button("🎯 Calcular Predicción", variant="primary", size="lg")
                out_pred = gr.Markdown()
                btn.click(
                    fn=predict_approval,
                    inputs=[inp_rep, inp_des],
                    outputs=[out_pred]
                )
        
        # TAB 3
                # TAB 3
        with gr.TabItem("🧠 Informe y Toma de Decisiones"):

            gr.Markdown("""
# 📋 Informe Analítico y Laboratorio de Decisiones

## 🚨 Problema de Investigación
En el sistema educativo, dos fenómenos afectan directamente los resultados:

- La **deserción escolar** (estudiantes que abandonan)
- La **reprobación académica** (estudiantes que no aprueban)

👉 Ambos reducen la tasa de aprobación, pero no necesariamente con la misma intensidad.

---

## ❓ Pregunta clave

### **¿Qué factor impacta más la aprobación educativa: la deserción o la reprobación?**

Este informe no solo responde esa pregunta,  
👉 te permite **experimentar con ella**.

---
## 🧪 Exploración Interactiva
Asume el rol de un tomador de decisiones y contrasta tu intuición con evidencia.
""")

            # -------------------------
            # QUIZ
            # -------------------------
            gr.Markdown("### 1️⃣ Validación de intuición")

            quiz = gr.Radio(
                choices=["Deserción", "Reprobación"],
                label="¿Cuál crees que tiene mayor impacto en la aprobación?"
            )

            quiz_btn = gr.Button("Analizar respuesta")
            quiz_out = gr.Markdown()

            def quiz_answer(q):
                if q is None:
                    return "⚠️ Selecciona una opción para continuar."

                return f"""
**Tu elección:** {q}

### 📊 Evidencia del modelo
👉 La **deserción** tiene un impacto mayor sobre la aprobación.

### 💡 Interpretación
Cuando un estudiante abandona:
- Sale completamente del sistema
- No tiene posibilidad de aprobar

En cambio, un estudiante que reprueba:
- Permanece en el sistema
- Puede recuperarse

**Conclusión:**  
Reducir la deserción genera un efecto más fuerte sobre los resultados educativos.
"""

            quiz_btn.click(fn=quiz_answer, inputs=quiz, outputs=quiz_out)

            # -------------------------
            # RESULTADOS MODELO
            # -------------------------
            gr.Markdown("""
---
## 📊 Resultados del Modelo

| Métrica | Valor | Interpretación |
|--------|------|----------------|
| R² Ajustado | 0.677 | Alta capacidad explicativa |
| MAE | 0.015 | Bajo error |
| Independencia | ✔ | Sin autocorrelación |
| Multicolinealidad | ✔ | Variables estables |

👉 El modelo es estadísticamente confiable para análisis estratégico.
""")

            # -------------------------
            # DECISION
            # -------------------------
            gr.Markdown("### 2️⃣ Simulación de decisión pública")

            decision = gr.Radio(
                choices=[
                    "Invertir en reducir deserción",
                    "Invertir en mejorar rendimiento",
                    "Dividir recursos"
                ],
                label="Si tuvieras recursos limitados, ¿dónde invertirías?"
            )

            decision_btn = gr.Button("Evaluar decisión")
            decision_out = gr.Markdown()

            def decision_logic(d):
                if d is None:
                    return "⚠️ Selecciona una opción."

                if "deserción" in d:
                    return """
🟢 **Decisión óptima**

Estás priorizando el factor con mayor impacto estructural.

✔ Mayor efecto en aprobación  
✔ Reduce pérdida total del sistema  
"""
                elif "Dividir" in d:
                    return """
🟡 **Decisión intermedia**

Equilibra esfuerzos, pero reduce eficiencia del impacto.

✔ Enfoque balanceado  
⚠ Menor efecto total  
"""
                else:
                    return """
🔴 **Decisión subóptima**

Se enfoca en un factor relevante, pero no prioritario.

⚠ Impacto limitado en mejora global  
"""

            decision_btn.click(fn=decision_logic, inputs=decision, outputs=decision_out)

            # -------------------------
            # ESCENARIOS
            # -------------------------
            gr.Markdown("### 3️⃣ Interpretación de escenarios")

            scenario = gr.Dropdown(
                choices=[
                    "Alta deserción",
                    "Alta reprobación",
                    "Ambos altos"
                ],
                label="Selecciona un contexto educativo"
            )

            scenario_out = gr.Markdown()

            def scenario_text(s):
                if s is None:
                    return "Selecciona un escenario."

                if s == "Alta deserción":
                    return """
🔴 **Escenario crítico**

- Pérdida estructural del sistema
- Impacto directo en aprobación

👉 Prioridad: retención escolar
"""
                elif s == "Alta reprobación":
                    return """
🟡 **Escenario académico**

- Problemas de aprendizaje
- Impacto moderado en aprobación

👉 Prioridad: refuerzo educativo
"""
                else:
                    return """
⚫ **Crisis estructural**

- Fallas simultáneas del sistema
- Máximo impacto negativo

👉 Requiere intervención integral
"""

            scenario.change(fn=scenario_text, inputs=scenario, outputs=scenario_out)

            # -------------------------
            # HALLAZGOS
            # -------------------------
            gr.Markdown("""
---
## 🔍 Hallazgos Clave del Análisis

- La **deserción es el factor más determinante**
- La **reprobación tiene impacto secundario**
- El sistema educativo pierde más por abandono que por bajo rendimiento
- Las decisiones de política deben priorizar impacto estructural

---
## 🧠 Insight Estratégico

👉 **No todos los problemas pesan igual**

Optimizar decisiones implica priorizar correctamente.

---
## 🎯 Conclusión General

El análisis confirma que:

- Mejorar notas es importante  
- Pero **evitar el abandono es crítico**

👉 **El sistema educativo falla más cuando pierde estudiantes que cuando estos reprueban.**

---
## 💥 Mensaje Final

La educación no se transforma solo mejorando resultados.  
Se transforma **evitando que los estudiantes desaparezcan del sistema**.
""")

            
if __name__ == "__main__":
    app.launch()