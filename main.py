import pandas as pd
import pickle as pkl

def sistema_inteligente(
    bill_length_mm,
    bill_depth_mm,
    flipper_length_mm,
    body_mass_g
    ):
    """
    Clasifica un pingüino basándose en reglas diseñadas por un humano.

    Parámetros:
    -----------
    bill_length_mm : float
      Longitud del pico en milímetros
    bill_depth_mm : float
      Profundidad del pico en milímetros
    flipper_length_mm : float
      Longitud de la aleta en milímetros
    body_mass_g : float
      Masa corporal en gramos

    Retorna:
    --------
    str : 'Adelie', 'Chinstrap', o 'Gentoo'
      Raza del pingüino
    """

    if flipper_length_mm > 200:
        if bill_depth_mm < 18:
            return "Gentoo"
        else:
            if bill_length_mm > 45:
                return "Chinstrap"
            else:
                return "Adelie"
    else:
        if bill_length_mm > 45:
            return "Chinstrap"
        else:
            return "Adelie"

def clasificador_humano(data):
    """
    Basado en el sistema inteligente, recibe la información general
    y predice la especie de cada pingüino.

    Parámetros:
    -----------
    data : pd.DataFrame
      Datos a predecir

    Retorna:
    --------
    predicciones_humano : list
      Predicciones por pingüino.
    """
    predicciones_humano = []

    for idx, row in data.iterrows():
        pred = sistema_inteligente(
            row['bill_length_mm'],
            row['bill_depth_mm'],
            row['flipper_length_mm'],
            row['body_mass_g']
        )

        predicciones_humano.append(pred)

    return predicciones_humano

def comparador_humano_vs_maquina(data, modelo_maquina, modelo_humano):
    """
    Devuelve un csv y lo guarda en memoria.

    Parámetros:
    -----------
    data : pd.DataFrame
      Datos a predecir
    modelo_maquina : sklearn.tree.DecisionTreeClassifier
      Modelo de árbol de decisión entrenado.
    modelo_humano : function
      Función que predice la especie de cada pingüino.

    Retorna:
    --------
    results : pd.DataFrame
      Predicciones por pingüino.
    """

    y_pred_maquina = modelo_maquina.predict(data)
    y_pred_humano = modelo_humano(data)
    
    results = pd.DataFrame({
        'y_pred_maquina': y_pred_maquina,
        'y_pred_humano': y_pred_humano
    })

    results.to_csv('results.csv', index=False)
    return results


path = "data.csv"
data = pd.read_csv(path)

X = data[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]

modelo_maquina_path = "modelo_ml.pkl"
with open("modelo_ml.pkl", 'rb') as f:
    modelo_maquina = pkl.load(f)
    
print(comparador_humano_vs_maquina(X, modelo_maquina, clasificador_humano))