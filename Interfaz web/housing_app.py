import gradio as gr 
import pandas as pd
import numpy as np
import os
import csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# CONFIGURACIÓN DE HISTORIAL
HISTORY_FILE = "historial_housing.csv"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, mode="r", newline="") as f:
            reader = csv.reader(f)
            return [row for row in reader]
    return []

def save_history_row(row):
    with open(HISTORY_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def clear_history_file():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)


# CARGA Y PREPARACIÓN DEL MODELO
df = pd.read_csv("housing.csv")
df = df.dropna().drop_duplicates()
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

feature_names = X.columns.tolist()


# FUNCIÓN DE PREDICCIÓN CON VALIDACIÓN
def predict_value(*inputs):
    # Validar que no todos los valores sean cero
    if all(float(val) == 0.0 for val in inputs):
        return "⚠️ Por favor ingresa valores distintos de cero. Todos los campos están en 0.", load_history()

    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    row = list(map(str, inputs)) + [f"{prediction:.2f}"]
    
    history = load_history()
    history.append(row)
    save_history_row(row)
    
    return f"Este distrito tiene un valor medio estimado de ${prediction:,.2f}", history

#
# FUNCIÓN PARA LIMPIAR ENTRADAS
def clear_inputs():
    clear_history_file()
    return [0.0] * len(feature_names), []


# INTERFAZ DE GRADIO
def housing_interface():
    history = load_history()  # Cargar historial actualizado

    with gr.Blocks() as demo:
        gr.Markdown("## 🏠 Predicción del Valor de Vivienda - California Housing")
        gr.Markdown("Completa las características del distrito para estimar el valor medio de sus viviendas:")

        # Distribuir inputs en 3 columnas
        with gr.Row():
            with gr.Column():
                inputs_col1 = [gr.Number(label=feature, value=0.0) for feature in feature_names[0::3]]
            with gr.Column():
                inputs_col2 = [gr.Number(label=feature, value=0.0) for feature in feature_names[1::3]]
            with gr.Column():
                inputs_col3 = [gr.Number(label=feature, value=0.0) for feature in feature_names[2::3]]

        # Unificar lista de inputs
        inputs = inputs_col1 + inputs_col2 + inputs_col3

        output = gr.Textbox(label="Resultado de la Predicción")
        submit_button = gr.Button("Predecir")
        clear_button = gr.Button("Limpiar")

        history_output = gr.Dataframe(
            headers=feature_names + ["Predicción"],
            label="Historial de Predicciones",
            interactive=False,
            value=history
        )

        submit_button.click(
            predict_value,
            inputs=inputs,
            outputs=[output, history_output]
        )

        clear_button.click(
            clear_inputs,
            inputs=[],
            outputs=inputs + [history_output]
        )

    return demo
