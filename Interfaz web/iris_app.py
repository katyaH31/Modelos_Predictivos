import gradio as gr
import csv
import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


# ---------CONFIGURACIÓN DE HISTORIAL----------
HISTORY_FILE = "historial.csv"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, mode="r", newline="") as f:
            reader = csv.reader(f)
            #Convirtiendo a float 
            return[
                [float(row[0]), float(row[1]), float(row[2]), float(row[3]), row[4]]
                for row in reader
            ]
    return []

def save_history_row(row):
    with open(HISTORY_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def clear_history_file():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)


# Cargar el conjunto de datos Iris
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# --------- Custom Logistic Regression (One-vs-Rest, Gradient Descent) ---------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, y, lr=0.1, epochs=1000):
    m, n = X.shape
    X = np.hstack((np.ones((m,1)), X))
    weights = np.zeros(n + 1)
    for _ in range(epochs):
        z = np.dot(X, weights)
        h = sigmoid(z)
        grad = np.dot(X.T, (h - y)) / m
        weights -= lr * grad
    return weights

def predict_custom(X, weights_all):
    X = np.hstack((np.ones((X.shape[0],1)), X))
    preds = np.array([sigmoid(np.dot(X, w)) for w in weights_all]).T
    return np.argmax(preds, axis=1)

# Escalar los datos para el modelo personalizado
scaler_custom = StandardScaler()
X_iris_scaled = scaler_custom.fit_transform(X_iris)

# Entrenar modelo personalizado One-vs-Rest
weights_all_custom = []
for class_label in np.unique(y_iris):
    y_binary = (y_iris == class_label).astype(int)
    weights = gradient_descent(X_iris_scaled, y_binary, lr=0.1, epochs=1000)
    weights_all_custom.append(weights)


# ------------------- Función de predicción -------------------
def predict(sepal_length, sepal_width, petal_length, petal_width):
    if all(float(val) == 0.0 for val in [sepal_length, sepal_width, petal_length, petal_width]):
        return "⚠️ Por favor ingresa valores válidos (mayores a cero).", load_history(), None
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler_custom.transform(input_data)
    pred_idx = predict_custom(input_scaled, weights_all_custom)[0]
    class_name = iris.target_names[pred_idx]
    result = f"Predicción (Regresión Logística Custom GD): {class_name}"

    image_path = f"./imagenes/{class_name}.jpg"
    row = [sepal_length, sepal_width, petal_length, petal_width, class_name]
    
    history = load_history()
    history.append(row)
    save_history_row(row)

    return result, history, image_path



# ------------------- Función para limpiar las entradas -------------------
def clear_inputs():
    clear_history_file()
    return 0.0, 0.0, 0.0, 0.0, [], None

# ------------------- Interfaz de Gradio -------------------
def iris_interface():
    history = load_history()
    with gr.Blocks() as demo:
        gr.Markdown("## Clasificación Iris con Regresión Logística (Descenso de Gradiente Custom)")
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                sepal_length = gr.Number(label="Sepal Length (cm)", value=0.0)
                sepal_width = gr.Number(label="Sepal Width (cm)", value=0.0)
                petal_length = gr.Number(label="Petal Length (cm)", value=0.0)
                petal_width = gr.Number(label="Petal Width (cm)", value=0.0)
                output = gr.Textbox(label="Output")
        with gr.Column(scale=4):
            with gr.Row():
                with gr.Column(scale=1):
                    image_output = gr.Image(
                                label="Imagen de la Flor",
                                type="filepath",
                                interactive=False,
                                height=267,
                                width=300,
                          )
                clear_button = gr.Button("Clear")
                submit_button = gr.Button("Submit")
                        

    history_output = gr.Dataframe(
        headers=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Predicción"],
        label="Historial de Predicciones",
        interactive=False,
        wrap=True,
        value=history 
    )
    
    submit_button.click(
        predict,
        inputs=[sepal_length, sepal_width, petal_length, petal_width],
        outputs=[output, history_output, image_output]
    )
    
    clear_button.click(
        clear_inputs,
        inputs=[],
        outputs=[sepal_length, sepal_width, petal_length, petal_width, history_output, image_output]
    )


# Lanzar la aplicación
    return demo
