# k Vecinos Más Cercanos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

## Paso 1: Carga de datos y exploración de su estructura

### Carga de datos

url= "https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/refs/heads/main/winequality-red.csv"
total_data=pd.read_csv(url, sep=";")
total_data.to_csv("../data/raw/total_data.csv")

### Exploración de su estructura

print(total_data.shape)
print(total_data.info())
total_data.head()
sorted(total_data["quality"].unique())

### Creación de la target: label

def label_creator(column):
    if column < 5:
        return 0
    elif column < 7:
        return 1
    else:
        return 2
    
total_data["label"] = total_data["quality"].apply(label_creator)

total_data["label"].value_counts().sort_index()

## Paso 2: Entrenamiento del modelo KNN

### Separación del dataset en test y train

X = total_data.drop(["quality","label"], axis=1)
y = total_data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train.head()

### Escalado Min-Max

min_max = MinMaxScaler()
min_max.fit(X_train)

with open("../models/min_max.pkl", "wb") as file:
    pickle.dump(min_max,file)

X_train_scal = min_max.transform(X_train)
X_train_scal = pd.DataFrame(X_train_scal, index = X_train.index, columns=X_train.columns)

X_test_scal = min_max.transform(X_test)
X_test_scal = pd.DataFrame(X_test_scal, index = X_test.index, columns=X_test.columns)

X_train_scal.to_excel("../data/processed/X_train_con_outliers_scal.xlsx", index = False)
X_test_scal.to_excel("../data/processed/X_test_con_outliers_scal.xlsx", index = False)

X_train_scal.head()

## Entrenamiento del modelo con k=5

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scal, y_train)
y_pred_train = knn_model.predict(X_train_scal)
y_pred_test = knn_model.predict(X_test_scal)

## Paso 3: Evaluación del rendimiento

### `accuracy_score`

print(f"Train: {accuracy_score(y_train, y_pred_train)}")
print(f"Test: {accuracy_score(y_test, y_pred_test)}")

### `confusion_matrix`

print(f"Train:\n{confusion_matrix(y_train, y_pred_train)}")
print(f"Test:\n{confusion_matrix(y_test, y_pred_test)}")

### `classification_report`

print(f"Train:\n{classification_report(y_train, y_pred_train)}")
print(f"Test:\n{classification_report(y_test, y_pred_test)}")

## Paso 4: Optimización de k

### Bucle para distintos valores de k

results = []

for k in range(1,31):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scal, y_train)
    y_k_pred = model.predict(X_test_scal)
    results.append(accuracy_score(y_test, y_k_pred))

results = np.array(results)
best_k_index = np.argmax(results)
best_k = best_k_index + 1
best_accuracy = results[best_k_index]

print(f"Mejor k: {best_k}")
print(f"Mejor accuracy: {best_accuracy:.4f}")

### Graficado de accuracy según k

plt.figure(figsize=(10, 5))
plt.plot(range(1, 31), results, marker='o')
plt.xlabel("Valor de k")
plt.ylabel("Accuracy")
plt.title("Accuracy según valor de k")
plt.grid(True)
plt.show()

### Entrenamiento final del modelo

final_knn_model = KNeighborsClassifier(n_neighbors=3)
final_knn_model.fit(X_train_scal, y_train)

### Guardado del modelo

with open("../models/knn_best_model_k3.pkl", "wb") as f:
    pickle.dump(final_knn_model, f)

## Función predictora de calidad del vino

def predict_wine_quality(nums):
    nums_scal = min_max.transform([nums])
    pred = final_knn_model.predict(nums_scal)
    if pred[0] == 0:
        return "Este vino puede que sea de calidad baja"
    elif pred[0] == 1:
        return "Este vino puede que sea de calidad media"
    else:
        return "Este vino puede que sea de calidad alta"