import gradio as gr
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import load_iris

# Cargar el conjunto de datos Iris para regresión logística
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Modelo de Regresión Lineal Multivariante
linear_model = LinearRegression()

# Ajustar el modelo de regresión lineal con datos ficticios (falta el ajuste real)
X_dummy = np.random.rand(100, 4)  # 4 características
y_dummy = np.random.rand(100)     # Variable objetivo
linear_model.fit(X_dummy, y_dummy)

# Modelo de Regresión Logística Multiclase (falta el ajuste real)
logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_iris, y_iris)

# Función para manejar las predicciones
def predict(sepal_length, sepal_width, petal_length, petal_width, model_type):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    if model_type == "Multivariate Linear Regression":
        prediction = linear_model.predict(input_data)
        return f"Linear Regression Prediction: {prediction[0]:.2f}"
    elif model_type == "Multiclass Logistic Regression":
        prediction = logistic_model.predict(input_data)
        class_name = iris.target_names[prediction[0]]
        return f"Logistic Regression Prediction: {class_name}"

# Función para limpiar las entradas
def clear_inputs():
    return 0.0, 0.0, 0.0, 0.0, "Multivariate Linear Regression"

# Interfaz de Gradio
with gr.Blocks() as demo:
    gr.Markdown("## Regression Application")
    
    with gr.Row():
        sepal_length = gr.Number(label="Sepal Length (cm)", value=0.0)
        sepal_width = gr.Number(label="Sepal Width (cm)", value=0.0)
        petal_length = gr.Number(label="Petal Length (cm)", value=0.0)
        petal_width = gr.Number(label="Petal Width (cm)", value=0.0)
    
    model_type = gr.Radio(
        ["Multivariate Linear Regression", "Multiclass Logistic Regression"],
        label="Select Model",
        value="Multivariate Linear Regression"
    )
    
    output = gr.Textbox(label="Output")
    
    with gr.Row():
        clear_button = gr.Button("Clear")
        submit_button = gr.Button("Submit")
    
    submit_button.click(
        predict,
        inputs=[sepal_length, sepal_width, petal_length, petal_width, model_type],
        outputs=output
    )
    clear_button.click(
        clear_inputs,
        inputs=[],
        outputs=[sepal_length, sepal_width, petal_length, petal_width, model_type]
    )

# Lanzar la aplicación
demo.launch()