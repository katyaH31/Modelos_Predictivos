import gradio as gr
from iris_app import iris_interface
from housing_app import housing_interface

with gr.Blocks() as main_demo:
    gr.Markdown("<h1 style='text-align: center;'>Menú de Interfaces de Machine Learning</h1>")
    gr.Markdown("<p style='text-align: center;'>Seleccione una pestaña de su preferencia:</p>")

    with gr.Tabs():
        with gr.TabItem("🪻 Regresión Logística Clasificación tipo de flor Iris"):
            iris_interface()  

        with gr.TabItem("🏘️ Regresión Multivariada de Predicción de Viviendas"):
            housing_interface()

main_demo.launch()
