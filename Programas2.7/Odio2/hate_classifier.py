import numpy as np
from gensim import corpora
from limpiador import Limpiador
from postagger import SpanishParser, traductor
from supervisado.Datos import Datos
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from collections import defaultdict







class Hate_classifier:

    paser = None
    limpiador = None
    dictionary = None
    clf = None
    vocab_features = None
    encAtributos = None

    def __init__(self):
        self.parser = SpanishParser(morphology_path="morfo/es-morphology2.txt",context_path="morfo/es-context2.txt",lexicon_path= "morfo/es-lexicon4.txt")
        self.limpiador =  Limpiador(None)
        aux_dic = corpora.Dictionary()
        aux_dic = aux_dic.load('tweets.dict')
        self.dictionary = defaultdict(lambda:-1)
        self.vocab_features = self.load_filtered_features()
        self.encAtributos = preprocessing.OneHotEncoder(sparse=False)
        for key in aux_dic.token2id.keys():
            self.dictionary[key] = aux_dic.token2id[key]

        self.clf = None


    def fit(self):
        datos = Datos(r"training_set/hate_training_set.data",True)
        dd = datos.datos[:,:-1]
        print(dd.shape)
        
        X = self.encAtributos.fit_transform(dd)
        Y = datos.datos[:,-1]
        print(X.shape)
        #self.clf = BernoulliNB(fit_prior=False,alpha=0.00006)
        self.clf = MultinomialNB(fit_prior=False,alpha=0.00006)
        pesos = map(lambda x: x+1.1 if x==1 else 1 ,Y)
        self.clf.fit(X, Y,sample_weight=pesos)


    def predict(self,tweet):
        text = self.limpiador.clean(tweet)
        cadena = self.parser.parse(text,tokenize=True)
        cadenas = cadena.split()[0]
        vector = []
        for term in cadenas:
            meta = traductor(term[1])
            text = term[0]
            if meta in ["Adjetivo","Verbo","Nombre"]:
                vector.append(self.dictionary[text])

        print(vector)
        binary_vector= map(lambda term_vocab: 1 if term_vocab in vector else 0,self.vocab_features)
        binary_vector = np.array(binary_vector)
        binary_vector = binary_vector.reshape(1,-1)
        print(binary_vector)

        X = self.encAtributos.transform(binary_vector)
        print(X.shape)


        print(self.clf.predict(X))
       
        
    ## Cargar los atributos previamente obtenidos
    def load_filtered_features(self):
        dictionary = corpora.Dictionary()
        dictionary = dictionary.load('tweets.dict')
        f = open("training_set/filtered_features.txt","r")
        line = f.readline()
        frags = line.split(",")
        frags = frags[:-1]
        result = [ dictionary.token2id[term] for term in frags]
        f.close()
        return result



if __name__ == "__main__":
    hcl = Hate_classifier()
    hcl.fit()
    hcl.predict("eres moro?")




