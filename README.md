
# Sistema de apoyo al diagnóstico basado en inteligencia artificial para la detección temprana de cáncer de tiroides mediante ecografía

## Descripción General
Este proyecto implementa un **sistema de apoyo al diagnóstico médico** basado en **inteligencia artificial**, diseñado para **detectar de forma temprana el cáncer de tiroides** mediante imágenes ecográficas.  
El objetivo principal es **reducir los errores humanos** y **democratizar el acceso al diagnóstico especializado**, especialmente en regiones con escasez de radiólogos o infraestructura médica.

El proyecto fue desarrollado en el contexto del *Workshop de Impacto Social y Responsabilidad Ética* en la **Universidad de Especialidades Espíritu Santo (UEES)**.

---

## Objetivos del Proyecto
- Implementar un modelo de **aprendizaje profundo (Deep Learning)** capaz de analizar imágenes ecográficas de la glándula tiroides.  
- Clasificar los nódulos según su **probabilidad de malignidad** y generar un **reporte automatizado** para apoyo diagnóstico.  
- Promover la **responsabilidad ética** y la **transparencia** en el desarrollo y uso de sistemas de IA médica.  

---

## Arquitectura del Sistema
El sistema se compone de los siguientes módulos:

1. **Preprocesamiento de Imágenes**
   - Filtrado y normalización de ecografías.
   - Anonimización y eliminación de metadatos sensibles.

2. **Modelo de IA**
   - Red neuronal convolucional (CNN) entrenada en conjuntos de datos clínicos.
   - Clasificación binaria: *benigno* / *sospechoso de malignidad*.

3. **Interfaz Médica (UI)**
   - Visualización de imágenes y mapas de calor (Explainable AI – XAI).
   - Generación automática de reportes de diagnóstico.

4. **Módulo de Seguridad y Ética**
   - Control de acceso basado en roles.
   - Auditorías de fairness, transparencia y privacidad.

## Análisis Técnico de la Arquitectura del Sistema 
##          Integrador30.ipynb

## 1. Visión General del Sistema
El sistema implementado en el cuaderno Integrador30.ipynb constituye una plataforma automatizada de diagnóstico asistido por IA para la detección de nódulos tiroideos (benignos o malignos) a partir de imágenes médicas ecográficas.
La arquitectura combina análisis exploratorio de datos (EDA), detección de anomalías estructurales en el dataset, data augmentation, y un modelo de Deep Learning basado en EfficientNetB0, diseñado para clasificación binaria supervisada.
El flujo general se compone de los siguientes módulos:
1.	Carga y estructuración del dataset.
2.	Preprocesamiento y etiquetado.
3.	Análisis EDA y detección de outliers.
4.	Configuración de capas de aumento de datos.
5.	Definición, compilación y entrenamiento del modelo EfficientNet.
6.	Evaluación del modelo y almacenamiento persistente.
7.	Función de predicción interactiva con imagen cargada por el médico.
8.	Interpretación automatizada de diagnóstico.

## 2. Arquitectura de Procesamiento y Estructura de Datos
2.1. Carga y Exploración Inicial
El sistema inicia con la lectura de imágenes desde el repositorio:
/content/drive/MyDrive/p_1_image
Mediante os.walk() se estructura un DataFrame que almacena los metadatos de cada archivo: ruta, nombre, longitud y etiqueta inferida desde el directorio padre (benign o malignant).
Se incluyen controles de integridad:
•	Eliminación de registros vacíos o inconsistentes.
•	Verificación del número de archivos por clase.
•	Cálculo de la longitud del archivo (file_len) como variable auxiliar para detectar anomalías.
## 3. Módulo de Detección de Anomalías
Se aplica Isolation Forest (sklearn.ensemble.IsolationForest) sobre la columna file_len con una contaminación del 5%.
El objetivo es detectar posibles corrupciones o inconsistencias en las imágenes cargadas.
iso = IsolationForest(contamination=0.05)
df["anomaly"] = iso.fit_predict(df[["file_len"]])
Para evitar errores de ejecución con datasets vacíos, se incluye la validación:
if len(df[["file_len"]]) > 0:
    df["anomaly"] = iso.fit_predict(df[["file_len"]])
else:
    df["anomaly"] = []

## 4. Análisis Exploratorio de Datos (EDA)
Incluye:
•	Distribución de clases (benigno vs maligno).
•	Histogramas de longitudes de archivos y resolución de imágenes.
•	Conteo de muestras eliminadas como anomalías.
•	Curva de correlación entre tamaño de archivo y clase.
Este bloque permite evaluar el balance de clases y la calidad del dataset antes del entrenamiento.

## 5. Módulo de Preprocesamiento y Data Augmentation
Se implementa un pipeline de transformación usando tf.keras.Sequential, aplicado directamente dentro del modelo para mantener la coherencia entre entrenamiento y predicción:
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
])
El preprocesamiento incluye:
•	Redimensionamiento: imágenes a 224x224.
•	Normalización: reescalado de píxeles a rango [0,1].
•	Batching: división de datos en lotes (batch size configurable).

## 6. Arquitectura del Modelo — EfficientNetB0
El modelo base proviene de tf.keras.applications.EfficientNetB0, preentrenado en ImageNet, con pesos ajustados mediante transfer learning.
Estructura:
•	Input Layer: imágenes RGB de 224x224x3.
•	Bloques convolucionales profundos de EfficientNet con técnicas de depthwise separable convolutions.
•	Global Average Pooling Layer: reduce la dimensionalidad manteniendo la información semántica global.
•	Dropout (0.3): regularización para evitar sobreajuste.
•	Dense Layer (1, sigmoid): salida binaria (benigno/maligno).
Configuración de entrenamiento:
final_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
El entrenamiento se realiza durante 6 épocas (ajustadas para eficiencia en Colab) con validación cruzada sobre un conjunto de testeo del 20%.


## 7. Evaluación y Persistencia del Modelo
Tras el entrenamiento, se registran:
•	Accuracy y loss de entrenamiento y validación.
•	Curvas de aprendizaje.
•	Matriz de confusión y reportes de clasificación (precision, recall, F1-score).
El modelo se almacena en formato moderno .keras:
final_model.save("/content/thyroid_effnet_prod.keras")

## 8. Predicción Interactiva del Médico (Diagnóstico Asistido)
El módulo final solicita al doctor especialista cargar una imagen ecográfica mediante un cuadro de diálogo interactivo:
predict_uploaded_image(final_model)
Esta función realiza:
1.	Carga y preprocesamiento automático de la imagen subida.
2.	Ejecución del modelo final_model para predicción.
3.	Obtención del score de probabilidad.
4.	Interpretación médica automatizada:
o	p < 0.5 → diagnóstico probable: nódulo benigno.
o	p ≥ 0.5 → diagnóstico probable: nódulo maligno (sujeto a confirmación médica).
Salida textual ejemplo:
Probabilidad de malignidad: 0.83
Diagnóstico asistido: Lesión sospechosa (Maligna probable)
Recomendación: Correlacionar con estudio citológico y control clínico.


## 9.Arquitectura General del Sistema
Diagrama lógico resumido:
┌─────────────────────────────────────────────────────────┐
│       SISTEMA DE DIAGNÓSTICO AUTOMATIZADO DE TIROIDES   │
├─────────────────────────────────────────────────────────
│ 1️⃣ Carga Dataset (Imágenes)                           	│
│ 2️⃣ Limpieza y Etiquetado Automático               		│
│ 3️⃣ Detección de Anomalías (Isolation Forest)         	│
│ 4️⃣ Análisis EDA y Balanceo de Clases                  	│
│ 5     Data Augmentation (TF Layers)                   	│
│ 6     Entrenamiento EfficientNetB0 (Fine-Tuning)      	│
│ 7️⃣ Evaluación Métricas y Persistencia .keras          	│
│ 8️⃣ Interfaz Predictiva: Imagen → Diagnóstico Médico    │
└─────────────────────────────────────────────-----------┘

## 10. Resumen Técnico
El sistema implementado en Integrador30.ipynb presenta una arquitectura modular, robusta y escalable para diagnóstico asistido por IA en imágenes tiroideas.
La combinación de preentrenamiento EfficientNet, detección de anomalías y pipeline automático de predicción permite obtener resultados clínicamente interpretables y reproducibles en un entorno de investigación médica.
•	Framework principal: TensorFlow 2.x / Keras 3
•	Backend: Google Colab (GPU)
•	Entrada: Imágenes ecográficas .jpg / .png
•	Salida: Probabilidad de malignidad + recomendación diagnóstica

---

## Stakeholders y Grupos Impactados
| Stakeholder | Beneficios | Riesgos | Nivel de Influencia |
|--------------|-------------|----------|---------------------|
| Médicos radiólogos | Apoyo diagnóstico, reducción de carga laboral | Dependencia tecnológica | Alto |
| Pacientes | Diagnóstico rápido y confiable | Riesgo de error o desconfianza en IA | Medio |
| Instituciones de salud | Eficiencia y prestigio | Costos iniciales de implementación | Alto |
| Desarrolladores e investigadores | Innovación y publicaciones | Riesgos éticos y legales | Medio |

> **Grupos vulnerables:** Pacientes de zonas rurales con limitado acceso a servicios médicos.

---

## Impacto Social y Ético

### Impactos Positivos
- **Mejora del diagnóstico temprano** → mayor tasa de supervivencia.
- **Acceso equitativo** a diagnósticos especializados mediante telemedicina.
- **Optimización de recursos hospitalarios** gracias a la automatización.

### Riesgos Éticos
1. **Sesgo algorítmico:** Pérdida de precisión en grupos subrepresentados.  
2. **Privacidad de datos médicos:** Exposición de información sensible.  
3. **Falta de explicabilidad:** Dificultad para interpretar decisiones del modelo.  
4. **Responsabilidad clínica:** Dilema sobre quién responde ante un error diagnóstico.

---

## Estrategias de Mitigación
- **Balanceo de datasets** y auditorías de fairness trimestrales.  
- **Anonimización y cifrado** de todos los datos utilizados.  
- **Implementación de Explainable AI (XAI)** para mejorar la confianza médica.  
- **Revisión clínica obligatoria** antes de emitir diagnósticos finales.  

---

## Framework de Responsabilidad
- **Desarrolladores:** Garantizar precisión y control de versiones.  
- **Data Scientists:** Asegurar calidad y equidad de datos.  
- **Líder del Proyecto:** Supervisar cumplimiento ético.  
- **Comité Ético:** Evaluar impactos clínicos y auditorías.  

Mecanismos de *accountability*: documentación, auditorías semestrales, y alertas automáticas de sesgo o *data drift*.

---

## Estructura del Repositorio
```
├── Workshop Impacto Social y Responsabilidad en Proyecto de IA.pdf
│   └─ Documento completo con análisis ético y técnico del sistema.
├── Workshop Impacto Social y Responsabilidad en Proyecto de IA.pptx
│   └─ Presentación ejecutiva (resumen visual del proyecto).
└── README.md
```

---

## Reflexión Final
> “El mayor reto no fue técnico, sino ético: equilibrar la precisión de la IA con la responsabilidad médica humana.”

El proyecto demuestra cómo la inteligencia artificial puede **mejorar la equidad en salud**, siempre que se apliquen **principios de transparencia, justicia y responsabilidad compartida**.

---

## Autores
- **Byron Piedra** 
- **Christian García S.** 

**Supervisora:** Ing. Gladys Villegas  
**Institución:** Universidad de Especialidades Espíritu Santo (UEES)  
**Fecha:** Octubre 2025
