# Python 3.5
# Juan Carlos Pereira y Ana Peraita
from abc import ABCMeta,abstractmethod
from numpy import random


class Particion():
  
  indicesTrain=[]
  indicesTest=[]
  
  def __init__(self):
    self.indicesTrain=[]
    self.indicesTest=[]
  def __str__(self):
     return "Indices a entrenar: " +  str(len(self.indicesTrain)) + " Indices Test: " + str(len(self.indicesTest))
#####################################################################################################

class EstrategiaParticionado(object):
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta
  nombreEstrategia="null"
  numeroParticiones=0
  particiones=[]
  
  
  
  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada estrategia concreta  
  def creaParticiones(self,datos,seed=None):
    pass

  

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):
  probTrain=0.8
  
  def __init__(self, n, probTrain=0.8):
      self.nombreEstrategia = "ValidacionSimple"
      self.numeroParticiones = n
      self.particiones = []
      self.probTrain=probTrain
  
  # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
  # Devuelve una lista de particiones (clase Particion)
  def creaParticiones(self,datos,seed=None):
    random.seed(seed)
    for i in range(self.numeroParticiones):
        particion = Particion()
        lista = random.permutation(datos)
        n_entreno = int(self.probTrain*len(lista))
        particion.indicesTrain = lista[:n_entreno]
        particion.indicesTest = lista[n_entreno:]
        self.particiones.append(particion)
    return self.particiones
      
#####################################################################################################      
class ValidacionCruzada(EstrategiaParticionado):
  
  def __init__(self, n):
      self.nombreEstrategia = "ValidacionCruzada"
      self.numeroParticiones = n
      self.particiones = []
  

  
  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones
  # y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  def creaParticiones(self,datos,seed=None):   
    random.seed(seed)
    lista = list(random.permutation(datos))
    for i in range(self.numeroParticiones):
        particion = Particion()
        index =int(len(lista)/self.numeroParticiones)
        sobrante=len(lista)-(index*self.numeroParticiones)

        particion.indicesTrain = lista[:index*i]
        if i!=(self.numeroParticiones-1):
            particion.indicesTest = lista[index*i:index*(i+1)]
            particion.indicesTrain = particion.indicesTrain + lista[index*(i+1):]
        else:
            particion.indicesTest = lista[index*i:]
        
        self.particiones.append(particion)
    return self.particiones
    
    
