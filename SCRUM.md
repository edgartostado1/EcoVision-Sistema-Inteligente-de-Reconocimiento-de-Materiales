# EcoVision - Planificación Scrum

## Información del Proyecto

| Dato | Detalle |
|------|---------|
| **Proyecto** | EcoVision: Sistema Inteligente de Reconocimiento de Materiales |
| **Metodología** | Scrum |
| **Duración total** | 6 semanas |
| **Sprints** | 4 sprints de desarrollo + 2 sprints de proyecto final |
| **Equipo** | 3 personas |

---

## Roles Scrum

| Rol | Responsable | Función |
|-----|-------------|---------|
| **Product Owner** | | Define prioridades del Product Backlog |
| **Scrum Master** | | Facilita el proceso, elimina obstáculos |
| **Developer** | | Desarrolla las funcionalidades |

---

## Product Backlog

| ID | Tarea | Prioridad | Sprint |
|----|-------|-----------|--------|
| 1 | Definir categorías de materiales | Alta | Sprint 1 |
| 2 | Crear estructura del proyecto | Alta | Sprint 1 |
| 3 | Implementar preprocesamiento de imágenes | Alta | Sprint 1 |
| 4 | Diseñar arquitectura del modelo CNN | Alta | Sprint 2 |
| 5 | Implementar red neuronal | Alta | Sprint 2 |
| 6 | Recolectar dataset de imágenes | Alta | Sprint 2 |
| 7 | Entrenar modelo | Alta | Sprint 3 |
| 8 | Evaluar y ajustar modelo | Media | Sprint 3 |
| 9 | Crear interfaz de usuario (Streamlit) | Alta | Sprint 3 |
| 10 | Implementar sistema de recomendaciones | Media | Sprint 4 |
| 11 | Pruebas finales | Alta | Sprint 4 |
| 12 | Documentación y presentación | Media | Sprint 4 |

---

## Sprint 1: Estructura Base (Semana 1)

**Objetivo:** Tener la base del proyecto funcionando

| Tarea | Estado | Responsable |
|-------|--------|-------------|
| Definir categorías de materiales | ✅ Completado | |
| Crear estructura de carpetas | ✅ Completado | |
| Implementar `procesamiento/imagen.py` | ✅ Completado | |
| Configurar `requirements.txt` | ✅ Completado | |
| Crear `README.md` | ✅ Completado | |

**Entregable:** Estructura del proyecto con preprocesamiento de imágenes

---

## Sprint 2: Modelo de Deep Learning (Semana 2)

**Objetivo:** Tener el modelo CNN implementado y listo para entrenar

| Tarea | Estado | Responsable |
|-------|--------|-------------|
| Diseñar arquitectura CNN | ✅ Completado | |
| Implementar `modelo/red_neuronal.py` | ✅ Completado | |
| Implementar `modelo/entrenamiento.py` | ✅ Completado | |
| Recolectar dataset de imágenes | ⏳ Pendiente | |
| Organizar dataset en train/val | ⏳ Pendiente | |

**Entregable:** Código del modelo listo para entrenamiento

---

## Sprint 3: Entrenamiento e Interfaz (Semana 3)

**Objetivo:** Modelo entrenado y interfaz funcional

| Tarea | Estado | Responsable |
|-------|--------|-------------|
| Entrenar modelo con dataset | ⏳ Pendiente | |
| Evaluar precisión del modelo | ⏳ Pendiente | |
| Ajustar hiperparámetros si es necesario | ⏳ Pendiente | |
| Crear `interfaz/app.py` (Streamlit) | ✅ Completado | |
| Probar interfaz con modelo entrenado | ⏳ Pendiente | |

**Entregable:** Modelo entrenado + interfaz web funcional

---

## Sprint 4: Mejoras y Características Adicionales (Semana 4)

**Objetivo:** Agregar funcionalidades extra y mejorar el modelo

| Tarea | Estado | Responsable |
|-------|--------|-------------|
| Implementar sistema de recomendaciones | ⏳ Pendiente | |
| Agregar data augmentation avanzado | ⏳ Pendiente | |
| Mejorar precisión del modelo | ⏳ Pendiente | |
| Optimizar rendimiento | ⏳ Pendiente | |

**Entregable:** Modelo mejorado con características adicionales

---

## Sprint 5: Proyecto Final - Parte 1 (Semana 5)

**Objetivo:** Integración y pruebas completas

| Tarea | Estado | Responsable |
|-------|--------|-------------|
| Integrar todos los componentes | ⏳ Pendiente | |
| Pruebas de usabilidad | ⏳ Pendiente | |
| Corregir errores encontrados | ⏳ Pendiente | |
| Optimizar código | ⏳ Pendiente | |

**Entregable:** Sistema integrado y probado

---

## Sprint 6: Proyecto Final - Parte 2 (Semana 6)

**Objetivo:** Documentación y presentación

| Tarea | Estado | Responsable |
|-------|--------|-------------|
| Actualizar README | ⏳ Pendiente | |
| Completar documentación técnica | ⏳ Pendiente | |
| Preparar presentación final | ⏳ Pendiente | |
| Ensayo de la presentación | ⏳ Pendiente | |

**Entregable:** Proyecto completo, documentado y presentado

---

## Reuniones Scrum

| Reunión | Frecuencia | Duración |
|---------|------------|----------|
| **Daily Standup** | Diario | 15 min |
| **Sprint Planning** | Inicio de sprint | 1-2 horas |
| **Sprint Review** | Fin de sprint | 1 hora |
| **Sprint Retrospective** | Fin de sprint | 1 hora |

---

## Definition of Done

Una tarea se considera completada cuando:

- [ ] El código está escrito y funciona
- [ ] Se hicieron pruebas básicas
- [ ] El código está en GitHub
- [ ] Se actualizó el estado en este documento

---

## Notas

- Actualizar este archivo al final de cada sprint
- Marcar las tareas con ✅ cuando estén completas
- Agregar nuevas tareas al Product Backlog si surgen
