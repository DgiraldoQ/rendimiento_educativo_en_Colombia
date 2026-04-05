# Predicción del Rendimiento Educativo en Colombia

## 🎯 Descripción del Proyecto
Este proyecto analiza los factores asociados al rendimiento educativo en Colombia, específicamente la **aprobación académica**, utilizando un enfoque de **regresión lineal múltiple validada**.

A diferencia de modelos tipo "caja negra", el objetivo es construir un modelo **interpretable, estadísticamente válido y útil para la toma de decisiones** en contextos educativos.

---

## 🧠 Problema
El sistema educativo colombiano enfrenta desafíos como:
- Alta **deserción escolar**
- **Reprobación académica**
- Desigualdades en el acceso a recursos

Comprender qué variables influyen en estos resultados es clave para diseñar políticas públicas efectivas.

---

## 🔍 Enfoque Metodológico

### 1. Análisis Exploratorio de Datos (EDA)
- Identificación de patrones y relaciones
- Análisis de correlaciones
- Visualización de variables clave

### 2. Modelado
Se construyó un modelo de:
- **Regresión Lineal Múltiple**

### 3. Validación de Supuestos
El modelo cumple con los supuestos clásicos:

- ✔️ Linealidad  
- ✔️ Independencia de errores  
- ✔️ Homocedasticidad  
- ✔️ Normalidad de residuos  
- ✔️ **Ausencia de multicolinealidad (evaluada con VIF)**  

### 4. Optimización del Modelo
- Transformación **Box-Cox**
- Eliminación de variables no significativas (p < 0.05)
- Reducción de multicolinealidad

---

## 📈 Resultados

| Métrica         | Valor  | Interpretación                         |
|----------------|--------|----------------------------------------|
| R² Ajustado     | 0.677  | Alta capacidad explicativa              |
| MAE             | 0.015  | Bajo error en las predicciones         |

🔎 El modelo logra un buen equilibrio entre:
- Capacidad explicativa  
- Simplicidad  
- Interpretabilidad  

---

## 🧠 Insights Clave

- Existe una relación significativa entre **deserción y aprobación**
- El rendimiento académico está influenciado por múltiples factores sistémicos
- La reducción de variables redundantes mejora la estabilidad del modelo

---

## 🚀 Aplicación

Este modelo puede utilizarse para:

- Identificar estudiantes en riesgo  
- Apoyar la toma de decisiones en instituciones educativas  
- Diseñar políticas públicas basadas en datos  

---

## 🛠️ Tecnologías Utilizadas

- Python  
- Pandas  
- NumPy  
- Statsmodels  
- Scikit-learn  
- Matplotlib / Seaborn  

---

## 📂 Estructura del Proyecto

├── data/ # Datos utilizados
├── notebooks/ # Análisis exploratorio y modelado
├── models/ # Modelos entrenados
├── app/ # Aplicación interactiva (opcional)
├── README.md # Documentación


---

##  Recursos

- 📓 Análisis en Kaggle:  
  https://www.kaggle.com/code/dgiraldoq/an-lisis-de-factores-asociados-a-la-educacion-co  

- 🌐 Demo interactiva:  
  https://huggingface.co/spaces/analitycs/edu-predict-colombia  

---

##  Conclusión

Este proyecto demuestra que es posible construir modelos predictivos que no solo sean precisos, sino también **interpretables y estadísticamente confiables**, lo cual es fundamental en contextos donde las decisiones impactan directamente a las personas.

---

## 👤 Autor

**Diego Giraldo**  
Data Analyst / Data Science Enthusiast  

---

##  Nota Final
En análisis de datos, no basta con predecir:  
es necesario **explicar, validar y generar impacto**.
