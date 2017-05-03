# -*- coding: utf-8 -*-

# Python 3.5 y 2.7
# Juan Carlos Pereira y Ana Peraita
import numpy as np

class Datos(object):
  
    supervisado=True
    TiposDeAtributos=('Continuo','Nominal')
    tipoAtributos=[]
    nombreAtributos=[]
    nominalAtributos=[]
    datos=np.array(())
  # Lista de diccionarios. Uno por cada atributo.
    diccionarios=[]
    medias=np.array(())
    desviaciones=np.array(())
 
# Procesar el fichero para asignar correctamente las variables supervisado, tipoAtributos, nombreAtributos,nominalAtributos, datos y diccionarios
    def __init__(self, nombreFichero,sup):
        self.datos=np.array(())
        self.tipoAtributos=[]
        self.nombreAtributos=[]
        self.nominalAtributos=[]
        self.diccionarios=[]
        self.desviaciones=np.array(())
        self.medias=np.array(())
        archivo = open(nombreFichero, "r")
        #Leemos el número de datos que contiene el fichero
        n = int(archivo.readline())
        #Leemos el nombre de los atributos de los datos
        names = archivo.readline().split(",")
        self.nombreAtributos = names
        #Leemos el tipo de atributos Continuo o Nominal
        tipos = archivo.readline().splitlines()[0]
        self.tipoAtributos = tipos.split(",")
        #Creamos un array verificando si un atributo es nominal o no
        self.nominalAtributos = list(map(lambda x: self.evaluateNominal(x), self.tipoAtributos))
        
        datos = np.empty([n, len(names)], list)
        #Almacenamos los datos del fichero en una matriz
        for i in range(n):
            linea = archivo.readline().splitlines()[0]
            datos[i,:] = linea.split(",")
        archivo.close()
            
        #Creamos un diccionario por cada atributo y lo poblamos si es nominal
        self.createDict(datos)
        self.createDatos(datos)
    
    def evaluateNominal(self,x):
        "Evalua si es nominal o continuo, y lanza una excepción si no pertenece a ninguno de estos casos"
        if x==self.TiposDeAtributos[1]:
            return True
        elif x==self.TiposDeAtributos[0]:
            return False
        else:
            raise ValueError('Wrong nominal value')
    
    def createDict(self, datos):
        "Crea un diccionario para cada columna en datos y lo puebla dependiendo de los elementos que haya en datos"
        for i in range(len(self.nominalAtributos)):
            #Comprueba si es nominal, y si lo es crea un diccionario para el atributo
            if self.nominalAtributos[i]:
                aux_dic = ({})
                conjunto = set([elem[i] for elem in datos])
                contador = 0
                for elem in sorted(list(conjunto)):
                    aux_dic[elem] = contador
                    contador +=1
                self.diccionarios.append(aux_dic)
            else:
                self.diccionarios.append({})
    def createDatos(self, preDatos):
        "Crea una matriz poblada con los datos del diccionario y la asigna a la variable datos"
        postDatos=[]
        for preDato in preDatos:
            postDato=[]
            i=0
            for elemDato in preDato:
                # Si el diccionario tiene elementos se sustituye el valor por su respectivo número
                # Si no el elemento se añade a la matriz
                if any(self.diccionarios[i]):
                    postDato.append(self.diccionarios[i][elemDato])
                else:
                    postDato.append(float(elemDato))
                i+=1
            postDatos.append(postDato)
        self.datos = np.array(postDatos)

  # Obtiene los datos para realizar el entrenamiento correspondiente
    def extraeDatos(self,idx):
        matriz = np.empty([len(idx), len(self.nombreAtributos)], float)
        j = 0
        for i in idx:
            matriz[j] = self.datos[i]
            j+=1
        return matriz
        
    #Se calculan las medias y desviaciones tipicas [:,:-1]
    def calcularMediasDesv(self,datostrain):
        self.desviaciones=np.array(())
        self.medias=np.array(())
        self.medias = np.mean(datostrain, axis=0)
        self.desviaciones = np.std(datostrain,axis=0)
        #print(self.medias, self.desviaciones)
        
     #Se aplica la normalizacion sobre los datos [:,:-1]
    def normalizarDatos(self,datos):
        for i in range(len(self.medias)):
            datos[:,i] = datos[:,i] -self.medias[i]
            if(self.desviaciones[i]!=0):
                datos[:,i] = datos[:,i]/self.desviaciones[i]
        return datos
        



  