import gradio as gr
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

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
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler_custom.transform(input_data)
    pred_idx = predict_custom(input_scaled, weights_all_custom)[0]
    class_name = iris.target_names[pred_idx]
    return f"Predicción (Regresión Logística Custom GD): {class_name}"

# ------------------- Función para limpiar las entradas -------------------
def clear_inputs():
    return 0.0, 0.0, 0.0, 0.0

# ------------------- Interfaz de Gradio -------------------
with gr.Blocks() as demo:
    gr.Markdown("## Clasificación Iris con Regresión Logística (Descenso de Gradiente Custom)")
    
    with gr.Row():
        sepal_length = gr.Number(label="Sepal Length (cm)", value=0.0)
        sepal_width = gr.Number(label="Sepal Width (cm)", value=0.0)
        petal_length = gr.Number(label="Petal Length (cm)", value=0.0)
        petal_width = gr.Number(label="Petal Width (cm)", value=0.0)
    
    output = gr.Textbox(label="Output")
    
    with gr.Row():
        clear_button = gr.Button("Clear")
        submit_button = gr.Button("Submit")
    
    submit_button.click(
        predict,
        inputs=[sepal_length, sepal_width, petal_length, petal_width],
        outputs=output
    )
    clear_button.click(
        clear_inputs,
        inputs=[],
        outputs=[sepal_length, sepal_width, petal_length, petal_width]
    )

# Lanzar la aplicación
demo.launch()