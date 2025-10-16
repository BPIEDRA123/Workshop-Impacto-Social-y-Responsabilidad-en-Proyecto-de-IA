
# Sistema de IA para Detección Temprana de Cáncer de Tiroides mediante Ecografía

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
