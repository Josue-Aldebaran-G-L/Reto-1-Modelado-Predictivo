import sklearn as skl
import pandas as pd

def sistema_inteligente(
    bill_length_mm,
    bill_depth_mm,
    flipper_length_mm,
    body_mass_g):
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

def comparador_humano_vs_maquina(X_test, y_test, modelo_maquina):
    y_pred_maquina = modelo_maquina.predict(X_test)
    acurracy_maquina = skl.metrics.acurracy_score(y_test, y_pred_maquina)

    y_pred_humano = clasificador_humano(X_test)
    acurracy_humano = skl.metrics.accuracy_score(y_test, y_pred_humano)

    return (acurracy_maquina, acurracy_humano)


path = ""
data = pd.read_csv(path)

X = data[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = data['species']

X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(
    X,
    y,
    test_size=0.2
    random_state=42
    stratify=y

)

modelo_maquina = skl.tree.DecisionTreeClassifier(random_state=42)
modelo_maquina.fit(X_train, y_train)

comparador_humano_vs_maquina(X_test, y_test, modelo_maquina)