#  **Grupo:** InfoTurins

**Integrantes**

| Nº | Nombre completo                           | Código       |
|----|-------------------------------------------|--------------|
| 1  | ANTHONY ALEXANDER CANALES MARTINEZ        | 00093122     |
| 2  | KATYA LISBETH HERRERA MOLINA              | 00188119     |
| 3  | DANIELA IVETTE REINA MANZANARES           | 00051316     |
| 4  | AXEL JARED HERNANDEZ SERVELLON            | 00145319     |
| 5  | JOSUE ALFREDO MEJIA URIAS                 | 00000921     |

## Parte 1: Documento con Investigación teórica
Contiene el informe final y los archivos fuente en LaTeX del estudio comparativo de dos métodos numéricos de optimización: Gradiente Descendente y Newton-Raphson.

### Cómo compilar el informe

1. Extrae el contenido del archivo `.zip`.
2. Abre el archivo `.tex` principal con Overleaf o TeXstudio.
3. Compila con **PDFLaTeX**.

---

## Parte 2: Modelado Predictivo con Descenso de Gradiente

##  Descripción

Este proyecto tiene como objetivo aplicar técnicas de aprendizaje automático supervisado a dos conjuntos de datos:

1. **Iris (UCI)** – Clasificación de especies de flores mediante:
   - Regresión Logística Multiclase (One-vs-Rest)
   - Regresión Lineal Multivariada

2. **Precios de Viviendas en California (Kaggle)** – Predicción del valor medio de las casas en función de distintas variables geográficas y demográficas, usando:
   - Regresión Lineal Multivariada con descenso de gradiente

Ambos modelos comparten una implementación común del algoritmo de **descenso de gradiente**, reforzando el entendimiento práctico de este método.

---

## Requisitos para ejecutar el proyecto Notebook de Jupyter en formato .ipynb

Asegúrate de tener instalado Python 3.7 o superior.

Puedes instalar los paquetes necesarios con:

```bash
pip install pandas matplotlib seaborn numpy
```
Si estás utilizando Jupyter Notebook o Google Colab, puedes ejecutar el siguiente comando directamente en una celda:
```bash
!pip install pandas matplotlib seaborn numpy
```

## Parte 3: INTERFAZ WEB 

## Requisitos para ejecutar el proyecto de Interfaz Web (LOCAL)

Puedes instalar los paquetes necesarios con:

```bash
pip install gradio pandas numpy scikit-learn
```
Para lanzar la interfaz web de forma local, ejecuta el siguiente comando:

```bash
python main.py
```


