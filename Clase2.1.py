#Variable de texto
mi_variable = "Hola mundo"
print(mi_variable)

#lista de numeros, siempre entre corchetes
mi_lista=[1,2,3,4,5]
print(mi_lista)

#diccionario, algunas columnas estas etiquetadas con todo un nombre y ponen un codigo
#para diccionario siempre llaves
#tipo de objeto que permite dar una etiqueta a un valor 
# , para separar : para asignar, clave seria hombre y el valor seria 1 
mi_diccionario = {"clave_1":"valor1","clave_2":"valor2","clave_3":"valor3"}
print (mi_diccionario)

##############################
#NUMERICA

#Asentuar los tipos de variables que existen: numericos, cadenas o booleano
#declaro un objeto o una variable de enteros
#voy a multiplicar 5 veces a mi valor es decir 5 veces 10
vector_enteros = [10]*5
print(vector_enteros)

#vector de flotantes o decimales
vector_flotantes = [3.14]*5
print(vector_flotantes)

#crear un diccionario que contenga los vectores
diccionario = {"entero" : vector_enteros, "decimales o flotante" : vector_flotantes, "complejo" : vector_flotantes}
print(diccionario)

#############################
#CADENAS
#Mensajes y salidas para interaccion con el usuario

cadena_simple = 'Hola mundo!'

# contiene dos variables de texto
cadena_doble = ["PYTHON es poderoso", "Me gusta mucho"]
print (cadena_doble)

########################
#Dataframe
#libreria pandas es la que nos ayuda a trabajar dataframe

#importamos que y como la vamos a llamar
import pandas as pd

#Crear un Dataframe con los datos de rendimiento en juegos

datos = {
    'Nombre' : ['Juan', 'María', 'Carlos','Ana'],
    'Juego 1 (puntos)' : [150, 180, 130, 200],
    'Juego 2 (puntos)' : [150, 180, 130, 200],
    'Juego 3 (puntos)' : [150, 180, 130, 200],
}

#como quiero que se llame mi tabla, = llamo a pandas. DataFrame y lo que cree 
df = pd.DataFrame(datos)

#Mostrar el DataFrame
print(df)



