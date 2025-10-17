
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

│       SISTEMA DE DIAGNÓSTICO AUTOMATIZADO DE TIROIDES   │

├─────────────────────────────────────────────────────────│

1️⃣ Carga Dataset (Imágenes) 

2️⃣ Limpieza y Etiquetado Automático

3️⃣ Detección de Anomalías (Isolation Forest) 

4️⃣ Análisis EDA y Balanceo de Clases 

5  Data Augmentation (TF Layers) 

6  Entrenamiento EfficientNetB0 (Fine-Tuning) 

7️⃣ Evaluación Métricas y Persistencia .keras

8️⃣ Interfaz Predictiva: Imagen → Diagnóstico Médico 




## 10. Resumen Técnico
El sistema implementado en Integrador30.ipynb presenta una arquitectura modular, robusta y escalable para diagnóstico asistido por IA en imágenes tiroideas.
La combinación de preentrenamiento EfficientNet, detección de anomalías y pipeline automático de predicción permite obtener resultados clínicamente interpretables y reproducibles en un entorno de investigación médica.

•	Framework principal: TensorFlow 2.x / Keras 3

•	Backend: Google Colab (GPU)

•	Entrada: Imágenes ecográficas .jpg / .png

•	Salida: Probabilidad de malignidad + recomendación diagnóstica

<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/6945b52d-eed8-4157-8d98-4b9298eb5a59" />








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
## 3. Evaluación de Impacto Social

3.1 Impactos Positivos
1. Mejora en la precisión diagnóstica
•	Descripción: El uso de modelos de aprendizaje profundo permite detectar patrones en ecografías de tiroides con un nivel de detalle que puede superar la percepción humana, reduciendo los falsos negativos en diagnósticos tempranos de cáncer.
•	Grupos beneficiados: Pacientes con sospecha de patologías tiroideas, radiólogos, endocrinólogos y hospitales públicos con recursos limitados.
•	Evidencia o justificación: Estudios recientes muestran que los sistemas basados en CNN alcanzan precisiones superiores al 90% en la clasificación de nódulos tiroideos (referencia: IEEE Transactions on Medical Imaging, 2023), mejorando la detección temprana y las tasas de supervivencia.
2. Democratización del acceso a diagnósticos especializados
•	Descripción: La integración del sistema en plataformas médicas digitales facilita que centros rurales o con baja disponibilidad de especialistas puedan acceder a un diagnóstico asistido por IA.
•	Grupos beneficiados: Comunidades rurales, hospitales públicos y clínicas de bajo presupuesto.
•	Evidencia o justificación: La implementación de herramientas IA en entornos con escasez de recursos ha demostrado reducir los tiempos de diagnóstico hasta en un 60%, aumentando la eficiencia del sistema de salud.
3. Optimización de recursos médicos y reducción de carga laboral
•	Descripción: El sistema puede preclasificar imágenes médicas, permitiendo que los radiólogos enfoquen su atención en casos complejos.
•	Grupos beneficiados: Profesionales de la salud, instituciones hospitalarias y sistemas de salud pública.
•	Evidencia o justificación: La automatización parcial de la lectura de imágenes permite procesar hasta diez veces más estudios en el mismo periodo, según métricas internas de proyectos piloto similares (OMS, 2022).


3.2 Impactos Negativos y Riesgos
1. Riesgo de sesgo algorítmico
•	Descripción del riesgo: Si el modelo fue entrenado con datos no representativos (por ejemplo, mayor proporción de un grupo étnico o rango etario), puede generar errores sistemáticos en diagnósticos fuera de ese rango.
•	Grupos vulnerables: Pacientes con características no representadas en el dataset.
•	Severidad: Alta
•	Probabilidad: Media
2. Dependencia excesiva de la tecnología
•	Descripción del riesgo: Médicos o instituciones pueden confiar ciegamente en los resultados de la IA sin una revisión clínica adecuada.
•	Grupos vulnerables: Pacientes en zonas donde el recurso humano especializado es limitado.
•	Severidad: Alta
•	Probabilidad: Media
3. Riesgos de privacidad y manejo de datos médicos
•	Descripción del riesgo: El almacenamiento y procesamiento de imágenes médicas podrían exponer datos sensibles si no se aplican protocolos de seguridad y anonimización adecuados.
•	Grupos vulnerables: Pacientes, instituciones médicas y sistemas de salud pública.
•	Severidad: Alta
•	Probabilidad: Media

3.3 Análisis de Equidad
Pregunta crítica: ¿El sistema reduce o aumenta desigualdades existentes?
•	Distribución de beneficios: Los mayores beneficios recaen en centros médicos con infraestructura tecnológica para implementar el modelo, aunque la intención es extenderlo a zonas con recursos limitados.
•	Distribución de riesgos: Los riesgos se concentran en poblaciones donde la IA opera con escasa supervisión médica o con datos insuficientes.
•	Acceso: Aún no todos los hospitales públicos cuentan con capacidad de cómputo ni conexión segura a la nube para ejecutar modelos de IA en tiempo real.
•	Barreras: Limitaciones tecnológicas, falta de capacitación en el personal médico, y resistencia a la adopción tecnológica en entornos tradicionales.

---
## Riesgos Éticos Específicos
## *Riesgo 1:* Sesgo en el Modelo por Datos Desbalanceados

•	Descripción:

El dataset está desbalanceado, con 377 imágenes "malignant" y 260 "benign", y no se encontró la carpeta "normal". Esto podría generar un modelo sesgado hacia la clase mayoritaria ("malignant"), afectando la precisión diagnóstica para casos benignos o normales.

•	Evidencia:

o	En la carga de datos, se reporta: malignant: 377 imágenes, benign: 260 imágenes, y  Carpeta no encontrada: /content/drive/MyDrive/p_1_image/normal.

o	El código incluye un balanceador avanzado (AdvancedBalancer), que trata de generar  efectividad.

•	Severidad: Alta

•	Grupo afectado: Pacientes con condiciones benignas o normales, que podrían recibir diagnósticos incorrectos.


## *Riesgo 2:* Privacidad de Datos Médicos Sensibles

•	Descripción:

El sistema procesa imágenes médicas de pacientes sin mencionar mecanismos de anonimización, consentimiento informado o protección contra re-identificación. Los datos se almacenan en Google Drive sin especificar medidas de seguridad avanzadas (ej.: cifrado).

•	Evidencia:

o	Uso de Google Colab y drive.mount() para acceder a imágenes.

o	Metadatos recopilados incluyen rutas de archivos y dimensiones originales, que podrían contener información identificable.

•	Severidad: Alta

•	Grupo afectado: Pacientes cuyas imágenes son utilizadas para entrenamiento o diagnóstico.

## *Riesgo 3:* Falta de Transparencia y Explicabilidad

•	Descripción:

El modelo CNN funciona como una "caja negra", sin integrar herramientas de explicabilidad (ej.: SHAP, LIME) para justificar decisiones. Los usuarios (médicos) podrían no entender cómo se llega a un diagnóstico.

•	Evidencia:

o	El código se centra en métricas de precisión y curvas de aprendizaje, pero no incluye interpretabilidad de predicciones.

o	La función generar_diagnostico_detallado() muestra probabilidades, pero no explica qué características de la imagen llevaron a la conclusión.

•	Severidad: Media

•	Grupo afectado: Médicos que usan el sistema para respaldar decisiones clínicas.

## *Riesgo 4:* Accountability en Caso de Error

•	Descripción:

No se define un protocolo claro de responsabilidad si el modelo falla en el diagnóstico (ej.: falsos negativos en cáncer). Tampoco hay mención de auditorías externas o procesos de apelación para pacientes afectados.

•	Evidencia:

o	El sistema genera un "informe médico" pero advierte: "Este diagnóstico es asistido por IA y debe ser validado por médico especialista", trasladando la responsabilidad al usuario.

o	No se documentan mecanismos para reportar errores o actualizar el modelo con retroalimentación.

•	Severidad: Alta

•	Grupo afectado: Pacientes sometidos a diagnósticos incorrectos y profesionales médicos que dependen del sistema.

## Estrategias de Mitigación
## *Recomendaciones:*

### *Riesgo 1:* Sesgo en el Modelo por Datos Desbalanceados

Riesgo a mitigar: Sesgo en el Modelo por Datos Desbalanceados

Estrategia propuesta: Balancear el dataset mediante técnicas de aumento de datos y muestreo, y evaluar métricas de fairness por clase.
Tipo:

•	✓ Técnica (cambios en modelo/datos)

Implementación:

1.	Recolectar imágenes de la clase faltante "normal" y aplicar técnicas de aumento de datos (e.g., rotaciones, GANs) para las clases minoritarias.

2.	Implementar muestreo con SMOTE y ajustar class weights en el entrenamiento del modelo.

3.	Evaluar métricas de fairness (ej.: igualdad de tasas de falsos positivos/negativos) usando Fairlearn o similares.

Cuándo: Antes del entrenamiento inicial y en cada actualización del dataset.

Responsable: Data scientist del equipo.

Efectividad esperada: Alta


### Riesgo 2:* Privacidad de Datos Médicos Sensibles

Riesgo a mitigar: Privacidad de Datos Médicos Sensibles

Estrategia propuesta: Implementar un protocolo de anonimización y seguridad de datos con cifrado y consentimiento explícito.
Tipo:

•	✓ Técnica (cambios en modelo/datos)

•	✓ Política (governance, procesos)

Implementación:

1.	Anonimizar metadatos (eliminar rutas, nombres) y aplicar técnicas de ofuscación a las imágenes.

2.	Cifrar datos en reposo (Google Drive) y en tránsito (protocolos HTTPS).

3.	Establecer políticas de consentimiento informado y acceso basado en roles.

Cuándo: Antes de la recolección de datos y de forma continua.

Responsable: Oficial de seguridad de datos (DPO) y equipo legal.

Efectividad esperada: Alta

### *Riesgo 3:* Falta de Transparencia y Explicabilidad

Riesgo a mitigar: Falta de Transparencia y Explicabilidad

Estrategia propuesta: Integrar herramientas de explicabilidad (ej.: grad-CAM) y documentación clara para médicos.
Tipo:

•	✓ Técnica (cambios en modelo/datos)

•	✓ Diseño (arquitectura, UX)

Implementación:

1.	Incorporar librerías como SHAP o grad-CAM para generar mapas de calor que resalten regiones relevantes en las imágenes.

2.	Diseñar informes de diagnóstico que incluyan visualizaciones y explicaciones de las predicciones.

3.	Documentar limitaciones del modelo en guías accesibles para usuarios médicos.

Cuándo: Durante el desarrollo y en cada iteración del modelo.

Responsable: Data scientist y diseñador de UX.

Efectividad esperada: Media-Alta

### *Riesgo 4:* Accountability en Caso de Error

Riesgo a mitigar: Accountability en Caso de Error

Estrategia propuesta: Establecer protocolos de supervisión humana, auditoría y procesos de apelación.
Tipo:

•	✓ Política (governance, procesos)

•	✓ Diseño (arquitectura, UX)

Implementación:

1.	Diseñar un flujo de trabajo donde el diagnóstico de la IA sea siempre validado por un médico antes de su uso clínico.

2.	Crear un sistema de registro de errores y retroalimentación para actualizar el modelo.

3.	Realizar auditorías trimestrales por terceros independientes para evaluar el rendimiento y impacto ético.

Cuándo: Continuo durante la operación del sistema.

Responsable: Equipo legal, médicos responsables y comité de ética.

Efectividad esperada: Alta

 

---

### Framework de Responsabilidad - Sistema de Análisis Tiroideo con IA

## 6.1 Cadena de Responsabilidad

Desarrolladores: Implementación técnica, seguridad del código

Data Scientists: Calidad de datos, fairness del modelo

Médicos Especialistas: Validación clínica, supervisión humana

Product Owner: Decisiones de diseño, trade-offs éticos

Oficial de Protección de Datos: Privacidad, consentimiento, seguridad

Organización: Governance, responsabilidad final


## 6.2 Mecanismos de Accountability

Documentación: Model card, datasheet, ethics statement, guía clínica

Supervisión: Revisión humana obligatoria, proceso de apelación, auditoría trimestral

Monitoreo: Métricas de fairness semanales, alertas por drift, reportes de incidentes

Transparencia: Consentimiento informado, explicaciones accesibles, documentación pública


## 6.3 Plan de Respuesta a Incidentes

Detección: Monitoreo automático, reportes de médicos, quejas de pacientes

Respuesta Inmediata: Suspender diagnóstico automático, notificar médicos en 2 horas, activar revisión humana

Investigación: Análisis root cause en 48 horas, auditoría completa en 7 días

Corrección: Retraining del modelo, compensación si aplica, actualización de procedimientos

Prevención: Incorporar lecciones aprendidas, actualizar protocolos, verificación antes de reimplementación

---
## 7. CONSIDERACIONES DE COMPLIANCE
### Regulaciones aplicables al proyecto de análisis de imágenes tiroideas con IA, considerando el contexto médico y los datos manejados.

## HIPAA (Health Insurance Portability and Accountability Act):

Requisitos: Protección de información de salud personal (PHI), incluyendo medidas administrativas, físicas y técnicas para garantizar la confidencialidad, integridad y disponibilidad de los datos. Notificación de violaciones de datos.

Cumplimiento actual: El proyecto utiliza imágenes médicas almacenadas en Google Drive, lo que podría no cumplir con los requisitos de seguridad y almacenamiento de PHI. No se mencionan medidas específicas de cifrado o acuerdos de socios comerciales (BAAs) con Google.

Cambios necesarios: Implementar medidas de seguridad acordes con HIPAA, como cifrado de datos en reposo y en tránsito, firmar un BAA con Google (si se usa Google Drive para PHI), y establecer políticas de acceso y auditoría.

## GDPR (General Data Protection Regulation):

Requisitos: Aplicable si se procesan datos de ciudadanos de la UE. Requiere consentimiento explícito, derecho al olvido, portabilidad de datos, y medidas de seguridad adecuadas.

Cumplimiento actual: No se menciona el consentimiento informado ni los derechos de los sujetos de datos. El proyecto recopila metadatos que podrían identificar a pacientes.

Cambios necesarios: Obtener consentimiento explícito para el uso de datos, implementar mecanismos para ejercer derechos de los sujetos, y asegurar que los datos se anonimizan adecuadamente.

## AI Act de la EU:

Requisitos: Clasificación del sistema de IA como de alto riesgo (ya que es un dispositivo médico). Requiere evaluación de conformidad, gestión de riesgos, datos de entrenamiento de alta calidad, documentación técnica y supervisión humana.

Cumplimiento actual: El proyecto no parece tener una evaluación de conformidad ni documentación técnica detallada. No se menciona la supervisión humana obligatoria en el flujo de trabajo.

Cambios necesarios: Realizar una evaluación de riesgo según el AI Act, documentar el sistema como dispositivo médico, implementar supervisión humana y garantizar la trazabilidad de las decisiones.

## Leyes anti-discriminación (ej. Equality Act en UK o Civil Rights Act en USA):

Requisitos: Evitar discriminación en base a características protegidas como raza, género, edad, etc.

Cumplimiento actual: El modelo podría tener sesgos debido al desbalance de datos (faltan imágenes "normales"). No se evalúan métricas de fairness por grupos demográficos.

Cambios necesarios: Recopilar datos balanceados por grupos demográficos, evaluar métricas de fairness, y mitigar sesgos mediante técnicas de preprocesamiento o postprocesamiento.

## FDA (U.S. Food and Drug Administration) para dispositivos médicos:

Requisitos: Si el sistema se usa para diagnóstico, podría requerir aprobación como dispositivo médico. Necesita validación clínica, documentación de rendimiento y gestión de riesgos.

Cumplimiento actual: El proyecto parece ser un prototipo y no tiene aprobación de la FDA.

Cambios necesarios: Realizar estudios clínicos para validar el rendimiento, documentar según los requisitos de la FDA y solicitar la aprobación correspondiente.

### Cambios necesarios:

Seguridad y privacidad: Implementar cifrado, acuerdos de socios comerciales para HIPAA, y políticas de acceso.

Consentimiento y derechos: Diseñar procesos para obtener consentimiento informado y permitir a los pacientes ejercer sus derechos bajo GDPR.

Evaluación de riesgos y documentación: Realizar una evaluación de conformidad según AI Act y FDA, documentar exhaustivamente el sistema.

Fairness: Balancear datos y evaluar métricas de fairness para evitar discriminación.

Estas consideraciones son críticas para desplegar el sistema en un entorno clínico real.


## Estructura del Repositorio
```
├── Workshop Impacto Social y Responsabilidad en Proyecto de IA.pdf
│   └─ Documento completo con análisis ético y técnico del sistema.
├── Workshop Impacto Social y Responsabilidad en Proyecto de IA.pptx
│   └─ Presentación ejecutiva (resumen visual del proyecto).
└── README.md
```

## Reflexión Final
“El mayor reto no fue técnico, sino ético: equilibrar la precisión de la IA con la responsabilidad médica humana.”

El proyecto demuestra cómo la inteligencia artificial puede **mejorar la equidad en salud**, siempre que se apliquen **principios de transparencia, justicia y responsabilidad compartida**.

---

## Autores
- **Ing. Byron Piedra** 
- **Ing. Christian García S., MBA** 

**Profesor:** Ing. Gladys Villegas  
**Institución:** Universidad de Especialidades Espíritu Santo (UEES)  
**Fecha:** Octubre 2025
