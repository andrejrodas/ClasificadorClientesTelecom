# %% [markdown]
# <a href="https://colab.research.google.com/github/AnIsAsPe/ClasificadorClientesTelecom/blob/master/Notebooks/Churn_Telecom.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# El conjunto de Datos fue obtenido de Kaggle [Telecom Churn Dataset](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets?datasetId=255093&sortBy=voteCount&select=churn-bigml-80.csv)

# %% [markdown]
# # 0.Bibliotecas y Funciones

# %%
# pip install dtreeviz  # versión 2.2.2

# %%
import numpy as np
import pandas as pd

# Funciones específicas de Sckit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc

# Para visualizar el árbol de decisión
import graphviz
import dtreeviz

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# %%
def describe_datos(df):
    """
    Función para describir un DataFrame de pandas.

    Devuelve:
    --------
    DataFrame
        Devuelve un DataFrame con la descripción de cada columna, incluyendo:
    (1) Tipo de columna, (2) Número de valores nulos, (3) Porcentaje de valores nulos
    (4) Número de valores únicos y (5) Valores únicos

    """

    unicos =[]
    for col in df:
        unicos.append(df[col].unique())
    unicos = pd.Series(unicos, index=df.columns)
    descripcion = pd.concat(
        [
            df.dtypes,
            df.isna().sum(),
            round(df.isna().sum()/len(df)*100, 1),
            df.nunique(),
            unicos
        ],
        axis=1
    )

    descripcion.columns = ['dtypes', 'null', '%null', 'nunique', 'unique']
    return descripcion

# %% [markdown]
# # 1.Lectura de datos
# Preguntas a responder:
# * ¿De qué tamaño es el conjunto de datos?
# * ¿Qué tipo de variable tenemos? ¿El tipo de variables detectado automáticamente es correcto o es necesario hacer alguna transformación?
# * ¿Existen o no valores nulos?
# * ¿Las clases están desbalanceadas? Si es así, en qué medida?

# %%
df = pd.read_csv('https://raw.githubusercontent.com/AnIsAsPe/ClasificadorClientesTelecom/master/Datos/churn_telecom_espa%C3%B1ol.csv')
print(df.shape)
df.head(3)

# %%
df.info()

# %%
describe_datos(df)

# %%
df['Codigo de area'] = df['Codigo de area'].astype('object')
df['Abandono'] = 1 - df['Abandono'].astype(int)

# %%
# Tasa de abandono antes de la división train/test
print("Distribución de Abandono en el conjunto completo:")
print(df['Abandono'].value_counts())
print("\nTasa de abandono (proporción):")
print(df['Abandono'].value_counts(normalize=True))
print(f"\nTasa de abandono: {df['Abandono'].mean():.4f} ({df['Abandono'].mean()*100:.2f}%)")

# %% [markdown]
# # 2.División Inicial Train/Test
# Dividimos el conjunto de datos ANTES de cualquier preprocesamiento para evitar obtener información del conjunto de validacion en el modelo (Data Leakage)

# %%
df_train, df_test = train_test_split(df,
                                       test_size=0.20,
                                       shuffle=True,
                                       stratify=df['Abandono'],
                                       random_state=8)

print("Registros en Train:", len(df_train))
print("Registros en Test:", len(df_test))

# %% [markdown]
# # 3.Exploración y preprocesamiento unicamente usando conjunto de entrenamiento

# %% [markdown]
# ## Exploración univariada y bivariada

# %%
df_train['Abandono'].value_counts(normalize=True)

# %%
numericas_cols = df_train.select_dtypes('number').columns

fig, subplot = plt.subplots(nrows=6, ncols=3, figsize=(8, 12))
subplot= subplot.flatten()  # facilita iterar por cada grafica

for i, column in enumerate(numericas_cols[:-1]):
    df_train.groupby('Abandono')[column].hist(alpha=0.4, bins=20, ax=subplot[i+1],
                                        density=True)    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
    subplot[i+1].set_title(column)
    subplot[i+1].legend(df_train['Abandono'].unique())

plt.tight_layout()
subplot[0].set_visible(False)
subplot[-1].set_visible(False)
subplot[-2].set_visible(False)
plt.show()

# %%
numericas_cols = df_train.select_dtypes('number').columns
fig, subplot = plt.subplots(nrows=6, ncols=3, figsize=(12, 18), sharex=True)
subplot= subplot.flatten()  # facilita iterar por cada grafica

for i, col in enumerate(numericas_cols[:-1]):
    sns.boxplot(y=col, x='Abandono', hue='Abandono', data=df_train,
                orient='v', palette='pastel', ax=subplot[i+1,])   # https://seaborn.pydata.org/generated/seaborn.boxplot.html
plt.tight_layout()
subplot[0].set_visible(False)
subplot[-1].set_visible(False)
subplot[-2].set_visible(False)
plt.show()

# %%
categoricas_cols = df_train.select_dtypes(object).columns

for col in categoricas_cols:
    counts = df_train[col].value_counts()
    order = counts.index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(15, 4)) # Create a figure with 1 row and 2 columns

    # Gráfico de Frecuencia
    counts.plot(kind='bar', ax=axes[0], alpha=.5)
    axes[0].set_title(f'Categorías de {col} ')
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Frecuencia')
    axes[0].tick_params(axis='x', rotation=45)


    # Gráfico de Barras Apilada
    crosstab= pd.crosstab(df_train[col], df_train['Abandono'], normalize ='index')
    crosstab = crosstab.reindex(order, axis=0)

    crosstab.plot(kind='bar', stacked=True, ax=axes[1], alpha=.5)
    axes[1].set_title(f'Relación entre {col} y Abandono')
    axes[1].set_xlabel(col)
    axes[1].set_ylabel('Proporción')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(title='Abandono', labels=['No Abandono', 'Abandono'])

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Transformación de Variables Categóricas

# %%
describe_datos(df_train[categoricas_cols])

# %% [markdown]
# Utilizando one-hot-encoding construimos un "encoder" para transformar las variables categoricas.

# %%
encoder = OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False)
encoder.fit(df_train[categoricas_cols]).set_output(transform='pandas')

# %%
def preprocesar_datos(df_input, encoder, categoricas_cols):
    # Aplicar en encoder(OHE) que aprendió con el conjunto de entrenamiento
    df_cat_encoded = encoder.transform(df_input[categoricas_cols])

    # Remplazar las variables categoricas por las nuevas columnas creadas mediante OHE
    df_final = df_input.drop(columns = categoricas_cols)
    df_final = pd.concat([df_final, df_cat_encoded], axis=1)

    # Separar la etiqueta de las características
    y = df_final['Abandono']
    X = df_final.drop('Abandono', axis=1)
    return X, y

# %%
X_train, y_train = preprocesar_datos(df_train, encoder, categoricas_cols)

print(f'Columnas en X_train: {X_train.shape[1]}\n')
for col in X_train.columns:
  print(col)

# %% [markdown]
# # 4.Entrenamiento del árbol de decisión

# %%
profundidad = None

# Entrenamiento del modelo.
clasificador = DecisionTreeClassifier(max_depth=profundidad,
                                      criterion="entropy",
                                      random_state=0)
clasificador.fit(X_train, y_train)
print("La profundida del árbol es: {}".format(clasificador.get_depth()))

# %% [markdown]
# # 5.Preparación de los datos de test

# %%
X_test, y_test = preprocesar_datos(df_test, encoder, categoricas_cols)
print(X_test.shape[1])
X_test.columns

# %% [markdown]
# # 6.Evaluación del modelo

# %%
# Predicción y evaluación sobre el conjunto de entrenamiento.
y_pred_train = clasificador.predict(X_train)
exactitud_train = accuracy_score(y_train, y_pred_train)*100
print("Exactitud conjunto de entrenamiento: {:4.2f}%".format(exactitud_train))

# Predicción y evaluación sobre el conjunto de prueba.
y_pred_test = clasificador.predict(X_test)
exactitud_test = accuracy_score(y_test, y_pred_test)*100
print("Exactitud conjunto de prueba: {:4.2f}%".format(exactitud_test),'\n')

# %%
# Entrenamiento y prueba del modelo con distintos niveles de profunidad

clf = {}
y_pred_train = {}
y_pred_test = {}
exactitud_train={}
exactitud_test = {}

for p in range(3,28):
    # Entrenamiento del modelo
    clf[p] = DecisionTreeClassifier(max_depth = p,
                               criterion = "entropy",
                               random_state = 0).fit(X_train, y_train)

    # Predicción y evaluación sobre el conjunto de entrenamiento
    y_pred_train[p] = clf[p].predict(X_train)
    exactitud_train[p] = accuracy_score(y_train, y_pred_train[p])*100

    # Predicción y evaluación sobre el conjunto de prueba
    y_pred_test[p] = clf[p].predict(X_test)
    exactitud_test[p] = accuracy_score(y_test, y_pred_test[p])*100

exactitud_df = pd.DataFrame( {'Entrenamiento':exactitud_train,
                           'Prueba': exactitud_test})

exactitud_df.plot.line(
    title ='Exactitud en la predicción según profundidad del árbol')
plt.show()

# %%
exactitud_df

# %%
profundidad_optima = exactitud_df['Prueba'].idxmax()
profundidad_optima

# %%
model = clf[profundidad_optima]

# %% [markdown]
# ## Visualización del árbol de decisión

# %%
# Exportar el árbol como archivo.dot

dot_data = export_graphviz(clf[9], feature_names=X_train.columns,
                           class_names=['No abandono','Abandono'],
                           max_depth = 9,
                           rounded = True,
                           filled = True,
                           )

graph =  graphviz.Source(dot_data, format='png')

graph.render('arbol_decision')              # guarda el archivo .dot y la gráfica png
graph

# %%
viz_tree = dtreeviz.model(model, X_train, y_train,
                target_name='Abandono',
                feature_names=X_train.columns,
                class_names=['No abandono','Abandono'])

v = viz_tree.view(fontname="monospace", scale=1.2)     # render para guardar en formato
v.save("viz_tree.svg")
v

# %% [markdown]
# ## Importancia de cada variable de acuerdo al modelo

# %%
importancia = pd.Series(model.feature_importances_,
                    index=X_train.columns.values)

importancia.sort_values().plot(kind = 'barh',figsize=(10, 10))
plt.show()

# %% [markdown]
# ## Matriz de confusión y minimización de errores "graves"

# %% [markdown]
# **Matriz de confusión**

# %%
#Matriz de confusión
data = {'y_Real':y_test, 'y_Prediccion':model.predict(X_test)}

df = pd.DataFrame(data)
confusion_matrix = pd.crosstab(df['y_Real'], df['y_Prediccion'],
                               rownames=['Real'], colnames=['Predicted'])
fig, ax = plt.subplots(figsize=(4,3))
sns.heatmap(confusion_matrix, annot=True, fmt='g')
plt.show()

# %%
model.predict(X_test)[0:20]

# %%
pd.DataFrame(model.predict_proba(X_test), columns=['No Abandono', 'Abandono']).sort_values('Abandono')

# %%
umbral = 0.15 # por arriba del cual se clasificaría como clase 1
prediccion_test = np.where( model.predict_proba(X_test)[:, 1] > umbral, 'Abandono', 'No Abandono')
prediccion_test[0:20]

# %%
umbral = 0.12 # por arriba del cual se clasificaría como clase 1
prediccion_test = np.where( model.predict_proba(X_test)[:, 1] > umbral, True, False)
data = {'y_Real':  y_test,
        'y_Prediccion': prediccion_test
        }

evaluacion_df = pd.DataFrame(data)
confusion_matrix = pd.crosstab(evaluacion_df['y_Real'], evaluacion_df['y_Prediccion'], rownames=['Real'], colnames=['Predicted'])
fig, ax = plt.subplots(figsize=(4,3))
sns.heatmap(confusion_matrix, annot=True, fmt='g')
plt.show()

# %%
from sklearn.metrics import RocCurveDisplay

# %%
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(6, 6))

RocCurveDisplay.from_estimator(model, X_test, y_test, name='Profundidad 6', ax=ax)

RocCurveDisplay.from_estimator(clf[5], X_test, y_test, name='Profundidad 5', ax=ax)

RocCurveDisplay.from_estimator(clf[4], X_test, y_test, name='Profundidad 4', ax=ax)

ax.set_title('Curvas ROC para Arboles de Decisión con distintas profundidades')

plt.show()

# %% [markdown]
# # Bosque Aleatorio

# %% [markdown]
# ## Entrenamiento y prueba

# %%
clf_rf = RandomForestClassifier(n_estimators=1000,  random_state =0,
                                criterion='entropy').fit(X_train, y_train)

y_train_pred = clf_rf.predict(X_train)
y_pred = clf_rf.predict(X_test)
print("Exactitud del modelo Bosque Aleatorio en el conjunto de entrenamiento: {:4.2f}%".format(accuracy_score(y_train, y_train_pred )*100))
print("Exactitud del modelo Bosque Aleatorio en el conjunto de prueba: {:4.2f}%".format(accuracy_score(y_test, y_pred)*100))

# %%
importances = clf_rf.feature_importances_
weights = pd.Series(importances,
                    index=X_train.columns.values)
weights.sort_values().plot(kind = 'barh',figsize=(10, 10))

# %%
pd.DataFrame(clf_rf.predict_proba(X_test),
             columns=['No Abandono', 'Abandono']).sort_values('Abandono')

# %%
umbral = 0.10 # por arriba del cual se clasificaría como clase 1
prediccion_test = np.where( clf_rf.predict_proba(X_test)[:, 1] > umbral, True, False)
data = {'y_Real':  y_test,
        'y_Prediccion': prediccion_test
        }

evaluacion_df = pd.DataFrame(data)
confusion_matrix = pd.crosstab(evaluacion_df['y_Real'], evaluacion_df['y_Prediccion'], rownames=['Real'], colnames=['Predicted'])
fig, ax = plt.subplots(figsize=(4,3))
sns.heatmap(confusion_matrix, annot=True, fmt='g')
plt.show()

# %% [markdown]
# # Recursos:
# 
# Sayash, Kapoor, y Narayanan Arvind. «Leakage and the reproducibility crisis n machine-learning-based science». Patterns 4, n.º 9 (2023). https://doi.org/10.1016/j.patter.2023.100804.

# %%

