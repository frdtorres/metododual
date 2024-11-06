import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from scipy.optimize import linprog

# Configuración de la página
st.set_page_config(
    page_title="Programación Lineal: Método de Dos Fases y Dualidad",
    page_icon="📊",
    layout="wide"
)

# CSS personalizado para una mejor presentación
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .example-box {
        background-color: #e0f7fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    h1, h2, h3 {
        color: #ff1493;
    }
    .highlight {
        background-color: #ffff99;
        padding: 0.5rem;
        border-radius: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)

# Configuración de navegación
st.sidebar.title("📚 Navegación del Curso")
section = st.sidebar.radio("Seleccionar Tema", [
    "🎯 Introducción",
    "🔍 Fase 1: Encontrar Factibilidad",
    "⚡ Fase 2: Optimización",
    "📝 Ejemplo Completo",
    "🔄 Conceptos de Dualidad",
    "⚖️ Interpretación Económica"
])

def create_example_problem():
    """Crea un ejemplo real para demostración"""
    st.markdown("""
    ### Problema de Asignación de Recursos en Ciencia de Datos

    Un equipo de ciencia de datos está implementando dos modelos de machine learning: Modelo A y Modelo B. Desean maximizar la precisión total.

    **Recursos por unidad:**
    - Modelo A: 3 horas de GPU, 5 GB de RAM
    - Modelo B: 4 horas de GPU, 2 GB de RAM

    **Recursos Disponibles:**
    - 100 horas de GPU
    - 80 GB de RAM

    **Precisión por unidad:**
    - Modelo A: 0.8
    - Modelo B: 0.9

    **Formulación del Problema:**
    ```
    Maximizar: Z = 0.8x₁ + 0.9x₂
    Sujeto a:
        3x₁ + 4x₂ ≤ 100  (restricción de GPU)
        5x₁ + 2x₂ ≤ 80   (restricción de RAM)
        x₁, x₂ ≥ 0
    ```
    """)

    return {
        'c': [-0.8, -0.9],  # Negativo porque scipy.optimize minimiza
        'A_ub': [[3, 4], [5, 2]],
        'b_ub': [100, 80],
        'bounds': [(0, None), (0, None)]
    }

def plot_feasible_region(constraints, bounds):
    """Dibuja la región factible para el ejemplo"""
    x1 = np.linspace(0, 50, 1000)

    # Restricción de GPU: 3x₁ + 4x₂ ≤ 100
    x2_gpu = (100 - 3*x1) / 4

    # Restricción de RAM: 5x₁ + 2x₂ ≤ 80
    x2_ram = (80 - 5*x1) / 2

    fig = go.Figure()

    # Agregar restricciones
    fig.add_trace(go.Scatter(x=x1, y=x2_gpu, name='Restricción de GPU',
                             line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(x=x1, y=x2_ram, name='Restricción de RAM',
                             line=dict(color='red', dash='dash')))

    # Rellenar región factible
    x1_feasible = []
    x2_feasible = []
    for x in x1:
        y_gpu = (100 - 3*x) / 4
        y_ram = (80 - 5*x) / 2
        y = min(y_gpu, y_ram)
        if y >= 0:
            x1_feasible.append(x)
            x2_feasible.append(y)

    fig.add_trace(go.Scatter(x=x1_feasible, y=x2_feasible, 
                             fill='tozeroy', fillcolor='rgba(0,176,246,0.2)',
                             name='Región Factible', line=dict(color='rgba(255,255,255,0)')))

    fig.update_layout(
        title="Visualización de la Región Factible",
        xaxis_title="Unidades del Modelo A (x₁)",
        yaxis_title="Unidades del Modelo B (x₂)",
        showlegend=True
    )

    return fig

if section == "🎯 Introducción":
    st.title("Método de Dos Fases y Dualidad en Programación Lineal")
    
    st.markdown("""
    ### Visión General
    El Método de Dos Fases es una herramienta poderosa para resolver problemas de programación lineal cuando:
    1. No tenemos una solución básica factible obvia
    2. El problema tiene restricciones de igualdad o ≥

    ### Conceptos Clave
    - **Fase 1**: Encontrar una solución básica factible inicial
    - **Fase 2**: Optimizar la función objetivo
    - **Variables Artificiales**: Utilizadas para encontrar factibilidad
    - **Solución Básica**: Puntos extremos de la región factible
    """)
    
    # Mostrar un ejemplo sencillo
    st.markdown("""
    <div class='example-box'>
    <h3>Ejemplo Sencillo</h3>
    Considera un problema de asignación de recursos en ciencia de datos con dos variables:
    - x₁: cantidad de modelos entrenados con algoritmo A
    - x₂: cantidad de modelos entrenados con algoritmo B

    Con restricciones:
    1. x₁ + x₂ = 10
    2. 2x₁ + x₂ ≥ 8
    3. x₁, x₂ ≥ 0

    Este problema requiere el método de dos fases debido a las restricciones de igualdad y ≥.
    </div>
    """, unsafe_allow_html=True)

elif section == "🔍 Fase 1: Encontrar Factibilidad":
    st.title("Fase 1: Encontrar la Factibilidad Inicial")
    
    problem = create_example_problem()
    
    st.markdown("""
    ### Proceso de la Fase 1
    1. Agregar variables artificiales para crear una solución básica factible inicial
    2. Crear una función objetivo auxiliar que minimice la suma de variables artificiales
    3. Resolver utilizando el método simplex
    4. Verificar si las variables artificiales son cero
    """)
    
    # Demostración interactiva
    st.plotly_chart(plot_feasible_region(problem['A_ub'], problem['bounds']))
    
    # Mostrar el proceso con ejemplo real
    st.markdown("""
    ### Ejemplo: Convertir Restricciones ≥
    Para la restricción: 2x₁ + x₂ ≥ 8
    1. Restar variable de holgura: 2x₁ + x₂ - s₁ = 8
    2. Agregar variable artificial: 2x₁ + x₂ - s₁ + A₁ = 8

    ### Función Objetivo de la Fase 1
    Minimizar Z = A₁ (suma de variables artificiales)
    """)

elif section == "⚡ Fase 2: Optimización":
    st.title("Fase 2: Optimización")
    
    problem = create_example_problem()
    
    # Resolver el problema
    result = linprog(problem['c'], A_ub=problem['A_ub'], b_ub=problem['b_ub'],
                     bounds=problem['bounds'], method='highs')
    
    st.markdown("""
    ### Proceso de la Fase 2
    1. Eliminar variables artificiales
    2. Usar la solución básica factible inicial de la Fase 1
    3. Optimizar la función objetivo original
    """)
    
    # Mostrar solución óptima
    st.markdown("### Solución Óptima")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Modelos A (x₁)", f"{result.x[0]:.1f}")
    with col2:
        st.metric("Modelos B (x₂)", f"{result.x[1]:.1f}")
    
    st.metric("Precisión Máxima", f"{-result.fun:.2f}")
    
elif section == "📝 Ejemplo Completo":
    st.title("Ejemplo Completo: Asignación de Recursos")
    
    problem = create_example_problem()
    
    # Visualización
    st.plotly_chart(plot_feasible_region(problem['A_ub'], problem['bounds']))
    
    # Solución paso a paso
    st.markdown("""
    ### Solución Paso a Paso
    
    1. **Problema Original**
    ```
    Maximizar: Z = 0.8x₁ + 0.9x₂
    Sujeto a:
        3x₁ + 4x₂ ≤ 100  (GPU)
        5x₁ + 2x₂ ≤ 80   (RAM)
        x₁, x₂ ≥ 0
    ```
    
    2. **Forma Estándar**
    ```
    Maximizar: Z = 0.8x₁ + 0.9x₂
    Sujeto a:
        3x₁ + 4x₂ + s₁ = 100
        5x₁ + 2x₂ + s₂ = 80
        x₁, x₂, s₁, s₂ ≥ 0
    ```
    """)
    
    # Sliders interactivos para explorar soluciones
    st.markdown("### Explora Diferentes Soluciones")
    x1_val = st.slider("Unidades del Modelo A (x₁)", 0, 50, 20)
    x2_val = st.slider("Unidades del Modelo B (x₂)", 0, 25, 10)
    
    # Calcular restricciones
    gpu_used = 3*x1_val + 4*x2_val
    ram_used = 5*x1_val + 2*x2_val
    accuracy = 0.8*x1_val + 0.9*x2_val
    
    # Mostrar resultados
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("GPU Utilizadas", f"{gpu_used}/100 horas")
    with col2:
        st.metric("RAM Utilizada", f"{ram_used}/80 GB")
    with col3:
        st.metric("Precisión Total", f"{accuracy}")
    
elif section == "🔄 Conceptos de Dualidad":
    st.title("Dualidad en Programación Lineal")
    
    st.markdown("""
    ### Relación Primal-Dual
    
    Para cada problema de programación lineal (primal), existe un problema dual correspondiente:
    
    **Problema Primal:**
    ```
    Maximizar: Z = 0.8x₁ + 0.9x₂
    Sujeto a:
        3x₁ + 4x₂ ≤ 100  (y₁)
        5x₁ + 2x₂ ≤ 80   (y₂)
        x₁, x₂ ≥ 0
    ```
    
    **Problema Dual:**
    ```
    Minimizar: W = 100y₁ + 80y₂
    Sujeto a:
        3y₁ + 5y₂ ≥ 0.8  (x₁)
        4y₁ + 2y₂ ≥ 0.9  (x₂)
        y₁, y₂ ≥ 0
    ```
    """)
    
    # Demostración interactiva de dualidad
    st.markdown("### Precios Sombra (Variables Duales)")
    recurso = st.selectbox("Seleccionar Recurso", ["GPU", "RAM"])
    
    if recurso == "GPU":
        precios_sombra = np.linspace(0, 1, 100)
        valor = 100 * precios_sombra
    else:
        precios_sombra = np.linspace(0, 1, 100)
        valor = 80 * precios_sombra
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=precios_sombra, y=valor, 
                             mode='lines', name='Valor del Recurso'))
    fig.update_layout(
        title=f"Valor del Recurso {recurso} vs Precio Sombra",
        xaxis_title="Precio Sombra",
        yaxis_title="Valor del Recurso"
    )
    st.plotly_chart(fig)
    
elif section == "⚖️ Interpretación Económica":
    st.title("Interpretación Económica y Aplicaciones")
    
    st.markdown("""
    ### Precios Sombra y Significado Económico
    
    En nuestro ejemplo de asignación de recursos:
    - Cada variable dual (y₁, y₂) representa el valor marginal de los recursos
    - Los precios sombra indican cuánto aumentaría la precisión por unidad adicional de recurso
    
    ### Aplicaciones Prácticas
    1. **Asignación de Recursos**
    2. **Decisiones de Presupuesto**
    3. **Análisis de Sensibilidad**
    """)
    
    # Análisis de sensibilidad interactivo
    st.markdown("### Herramienta de Análisis de Sensibilidad")
    cambio_recurso = st.slider("Cambio en Disponibilidad de Recursos (%)", -20, 20, 0)
    
    # Calcular impacto
    precision_base = -problem['c'][0]*result.x[0] - problem['c'][1]*result.x[1]
    sensibilidad = 0.1    # Ejemplo de factor de sensibilidad
    nueva_precision = precision_base * (1 + cambio_recurso * sensibilidad / 100)
    
    st.metric("Impacto en la Precisión", 
              f"{nueva_precision:.2f}", 
              f"{cambio_recurso}% cambio en recursos")
    
if __name__ == "__main__":
    st.sidebar.markdown("""
    ### Acerca de esta Aplicación
    Esta herramienta educativa ayuda a entender:
    - Método de Dos Fases
    - Programación Lineal
    - Teoría de Dualidad
    - Aplicaciones en Ciencia de Datos
    
    Creado con fines educativos.
    """)
