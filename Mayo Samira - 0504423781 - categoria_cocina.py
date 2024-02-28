###### Practica - Curso ERGOSTATS

## Samira Mayo 

######################################################################
# IMPORTAMOS LAS BIBLIOTECAS QUE VAMOS A USAR EN EL TRABAJO 

# Se utiliza para realizar operaciones numéricas eficientes.
import numpy as np
# Trabaja con conjuntos de datos estructurados.
import pandas as pd
# Funciones para dividir conjuntos de datos y realizar validación cruzada.
from sklearn.model_selection import train_test_split, KFold
# Preprocesa nuestros datos antes de entrenar modelos de aprendizaje automático.
from sklearn.preprocessing import StandardScaler
# Proporciona métricas para evaluar el rendimiento de nuestros modelos.
from sklearn.metrics import accuracy_score
# Permite realizar análisis estadísticos más detallados y estimación de modelos.
import statsmodels.api as sm
# Ayuda a visualizar nuestros datos y resultados.
import matplotlib.pyplot as plt

######################################################################
# DATOS 

# Importamos la base de datos con la que trabajaremos 
datos = pd.read_csv("data/sample_endi_model_10p.txt", sep=";")
print(datos)

# Limpiamos la base, esto para facilidad de trabajar con datos unicamente necesarios 
datos = datos[~datos["categoria_cocina"].isna()]

# Suponiendo que 'datos' es tu DataFrame y 'variable' es el nombre de la columna
etiquetas_cocina = datos['etiquetas_cocina'].unique()
print(etiquetas_cocina)


cocina_hombre = datos[(datos['sexo'] == 'Hombre') & (datos['categoria_cocina'].isin(['Gas/Electricidad','Leña o carbón', 'No cocina']))]

# Calcular el conteo de niños por cada categoría de la variable 

conteo = cocina_hombre['categoria_cocina'].value_counts() 
print("Conteo de niños por categoría de 'categoria_cocina':")
print(conteo)

##################################################################################
# EJERCICIO 2

# Limpiamos la base
variables = ['sexo', 'etnia', 'region', 'quintil', 'condicion_empleo', 'categoria_cocina']
baselimpia = datos.dropna(subset=variables)

# Comprobar si hay valores no finitos después de la eliminación
print("Número de valores no finitos después de la eliminación:")
print(baselimpia.isna().sum())

# Convertir la variable categórica en binaria para poder trabajar con logit
baselimpia['cocina'] = baselimpia['categoria_cocina'].apply(lambda x: 1 if x == 'Gas/Electricidad' else 0)

# Filtrar los datos para incluir solo niños con sexo masculino y que tengan valores válidos 
hcocina = baselimpia[(baselimpia['sexo'] == 'Hombre') & (baselimpia['cocina'] == 1)]

# Seleccionar las variables relevantes
varmodelo = ['sexo', 'etnia', 'region', 'n_hijos', 'condicion_empleo', 'cocina']

# Filtrar los datos para las variables seleccionadas y eliminar filas con valores nulos en esas variables
for i in varmodelo:
    hcocina = hcocina[~hcocina[i].isna()]

# Agrupar los datos por sexo y tipo de cocina y contar el número de niños en cada grupo
conteo_ninos_por_categoria_cocina = hcocina.groupby(["sexo", "cocina"]).size()
print("Conteo de niños por categoría de 'serv_hig':")
print(conteo_ninos_por_categoria_cocina)

# Definir las variables categóricas y numéricas
variables_categoricas = ['sexo', 'etnia', 'region', 'quintil', 'condicion_empleo']
variables_numericas = ['n_hijos']

# Crear un transformador para estandarizar las variables numéricas
transformador = StandardScaler()

# Crear una copia de los datos originales
datos_escalados = baselimpia.copy()

# Estandarizar las variables numéricas
datos_escalados[variables_numericas] = transformador.fit_transform(datos_escalados[variables_numericas])

# Convertir las variables categóricas en variables dummy
dummies = pd.get_dummies(datos_escalados, columns=variables_categoricas, drop_first=True)
# Seleccionar las variables predictoras (X) y la variable objetivo (y)
X = dummies[['n_hijos', 'sexo_Mujer', 
                   'condicion_empleo_Empleada', 'condicion_empleo_Inactiva', 'condicion_empleo_Menor a 15 años']]
y = dummies["cocina"]

# Definir los pesos asociados a cada observación
weights = dummies['fexp_nino']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

# Asegurar que todas las variables sean numéricas
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')
y_train = y_train.apply(pd.to_numeric, errors='coerce')

# Convertir las variables a tipo entero
variables = X_train.columns
for i in variables:
    X_train[i] = X_train[i].astype(int)
    X_test[i] = X_test[i].astype(int)

y_train = y_train.astype(int)

# Ajustar el modelo de regresión logística
modelo = sm.Logit(y_train, X_train)
result = modelo.fit()
print(result.summary())

# Interpretación
# El coeficiente asociado al numero de hijos es -0.2202. Manteniendo todas las demás variables constantes, por cada unidad adicional en el número 
# de hijos, se espera que la probabilidad de tener cocina eléctrica/gas disminuya en un factor de exp(-0.2202), entonces tener más hijos 
# está asociado con una menor probabilidad de tener cocina eléctrica/gas, sin embargo esta variable no es significativa.

# El coeficiente asociado a las mujeres es 2.4031. Esto indica que las mujeres tienen una probabilidad mucho mayor de tener cocina eléctrica/gas
# en comparación con los hombres. Además, esta variable es significcativa por lo que sugiere una fuerte asociación entre el sexo y el tipo de cocina.

# El coeficiente asociada a la condición empleada es 2.5730. Indica que las personas empleadas tienen una probabilidad mucho mayor de tener cocina 
# eléctrica/gas en comparación con otras condiciones laborales. Esta variable tambien es significativa.

#############################################################################################################

# EJERCICIO 3
# Ajustar el modelo de regresión logística con regularización ridge
from sklearn.linear_model import LogisticRegressionCV

# Crear un modelo de regresión logística con regularización ridge
model = LogisticRegressionCV(cv=5, penalty='l2')

# Ajustar el modelo a los datos de entrenamiento
model.fit(X_train, y_train)

# Imprimir los coeficientes
print("Coeficientes:", model.coef_)
print("Intercepto:", model.intercept_)

# Calcular la precisión del modelo en los datos de prueba
accuracy = model.score(X_test, y_test)
print("Precisión del modelo:", accuracy)

# Calcular la precisión promedio de la validación cruzada
mean_accuracy = np.mean(accuracy)
print(f"Precisión promedio de validación cruzada: {mean_accuracy}")
# Calcular la precisión promedio
precision_promedio = np.mean(accuracy)

# Crear el histograma
plt.hist(accuracy, bins=30, edgecolor='black')

# Añadir una línea vertical en la precisión promedio
plt.axvline(precision_promedio, color='red', linestyle='dashed', linewidth=2)

# Añadir un texto que muestre la precisión promedio
plt.text(precision_promedio - 0.1, plt.ylim()[1] - 0.1, f'Precisión promedio: {precision_promedio:.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

# Configurar el título y etiquetas de los ejes
plt.title('Histograma de Accuracy Scores')
plt.xlabel('Accuracy Score')
plt.ylabel('Frecuencia')

# Ajustar los márgenes
plt.tight_layout()

# Mostrar el histograma
plt.show()

#####################################################################33
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import statsmodels.api as sm

# Definir las variables categóricas y numéricas
variables_categoricas = ['sexo', 'region', 'condicion_empleo']
variables_numericas = ['n_hijos']

# Crear un transformador para estandarizar las variables numéricas
transformador = StandardScaler()

# Crear una copia de los datos originales
datos_escalados = baselimpia.copy()

# Estandarizar las variables numéricas
datos_escalados[variables_numericas] = transformador.fit_transform(datos_escalados[variables_numericas])

# Convertir las variables categóricas en variables dummy
dummies = pd.get_dummies(datos_escalados, columns=variables_categoricas, drop_first=True)

# Seleccionar las variables predictoras (X) y la variable objetivo (y)
X = dummies[['n_hijos', 'sexo_Mujer', 'condicion_empleo_Empleada', 'condicion_empleo_Inactiva', 'condicion_empleo_Menor a 15 años']]
y = dummies["cocina"]

# Definir los pesos asociados a cada observación
weights = dummies['fexp_nino']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

# Asegurar que todas las variables sean numéricas
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')
y_train = y_train.apply(pd.to_numeric, errors='coerce')

# Convertir las variables a tipo entero
variables = X_train.columns
for i in variables:
    X_train[i] = X_train[i].astype(int)
    X_test[i] = X_test[i].astype(int)

y_train = y_train.astype(int)

# Definir el número de folds para la validación cruzada
kf = KFold(n_splits=100)
accuracy_scores = []  # Lista para almacenar los puntajes de precisión de cada fold
df_params = pd.DataFrame()  # DataFrame para almacenar los coeficientes estimados en cada fold

# Iterar sobre cada fold
for train_index, test_index in kf.split(X_train):
    # Dividir los datos en conjuntos de entrenamiento y prueba para este fold
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    weights_train_fold, weights_test_fold = weights_train.iloc[train_index], weights_train.iloc[test_index]
    
    # Ajustar un modelo de regresión logística en el conjunto de entrenamiento de este fold
    log_reg = sm.Logit(y_train_fold, X_train_fold)
    result_reg = log_reg.fit()
    
    # Extraer los coeficientes y organizarlos en un DataFrame
    coeficientes = result_reg.params
    df_coeficientes = pd.DataFrame(coeficientes).reset_index()
    df_coeficientes.columns = ['Variable', 'Coeficiente']
    df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
    df_pivot.reset_index(drop=True, inplace=True)
    
    # Realizar predicciones en el conjunto de prueba de este fold
    predictions = result_reg.predict(X_test_fold)
    predictions = (predictions >= 0.5).astype(int)
    
    # Calcular la precisión del modelo en el conjunto de prueba de este fold
    accuracy = accuracy_score(y_test_fold, predictions)
    accuracy_scores.append(accuracy)
    
    # Concatenar los coeficientes estimados en este fold en el DataFrame principal
    df_params = pd.concat([df_params, df_pivot], ignore_index=True)

# Calcular la precisión promedio del modelo
mean_accuracy = np.mean(accuracy_scores)
print("Precisión promedio del modelo:", mean_accuracy)

# Crear el histograma de los coeficientes para la variable "n_hijos"
plt.hist(df_params["n_hijos"], bins=30, edgecolor='black')

# Calcular la media de los coeficientes para la variable "n_hijos"
media_coeficientes_n_hijos = np.mean(df_params["n_hijos"])

# Añadir una línea vertical en la media de los coeficientes
plt.axvline(media_coeficientes_n_hijos, color='red', linestyle='dashed', linewidth=2)

# Añadir un texto que muestre la media de los coeficientes
plt.text(media_coeficientes_n_hijos - 0.1, plt.ylim()[1] - 0.1, f'Media de los coeficientes: {media_coeficientes_n_hijos:.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

# Configurar título y etiquetas de los ejes
plt.title('Histograma de Beta (N Hijos)')
plt.xlabel('Valor del parámetro')
plt.ylabel('Frecuencia')

# Ajustar los márgenes
plt.tight_layout()

# Mostrar el histograma
plt.show()
