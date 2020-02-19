import math
import random

import matplotlib as mpl
import matplotlib.animation
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

column_names = [
    "age_of_patient", "year_of_operation", "axillary_nodes_detected", "survival_status"
]
target_name = "survival_status"

csv = pd.read_csv("haberman.data", names=column_names)

target_data = csv.iloc[:, -1]
fields_data = csv.iloc[:, :-1]

haberman_dataframe = pd.DataFrame(fields_data, columns=column_names[:-1])
haberman_dataframe.head(10)

sp = pd.plotting.scatter_matrix(haberman_dataframe, c=target_data, figsize=(
    15, 15), marker='D', hist_kwds={'bins': 20}, s=60, alpha=.8)

sns.pairplot(data=csv, hue=target_name, markers=[
             "o", "D"], vars=column_names[:-1])

corr = csv.corr()
sns.heatmap(corr)

sns.lmplot(x="axillary_nodes_detected", y="age_of_patient",
           data=csv, hue=target_name, fit_reg=True)

sns.boxplot(data=csv, orient='h')

X_train, X_test, y_train, y_test = train_test_split(fields_data, target_data, stratify=target_data,
                                                    random_state=42, test_size=0.1)


def print_rendimiento(entrenamiento, prueba, total):
    print("Rendimiento en el conjunto de entrenamiento: ", entrenamiento)
    print("Rendimiento en el conjunto de prueba: ", prueba)
    print("Rendimiento en el conjunto total: ", total)


def generar_modelo(solver="lbfgs", f_activation="logistic", capas=(20,), aprendizaje=0.0001, batch="auto", max_iter=200):
    mlp_modelo = MLPClassifier(
        solver=solver, random_state=0, hidden_layer_sizes=capas, activation=f_activation, alpha=aprendizaje, batch_size=batch, max_iter=max_iter)
    mlp_modelo.fit(X_train, y_train)
    print_rendimiento(
        mlp_modelo.score(X_train, y_train),
        mlp_modelo.score(X_test, y_test),
        mlp_modelo.score(fields_data, target_data)
    )
    return mlp_modelo


def neuronas_capa_oculta(neuronas_entrada: int, neuronas_salida: int):
    return math.pow(neuronas_entrada * neuronas_salida, 1/2)


possible_output = 2
neuronas_ocultas = neuronas_capa_oculta(len(column_names) - 1, possible_output)

generar_modelo()
