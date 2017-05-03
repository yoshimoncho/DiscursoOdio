import numpy as np
import pickle
from gensim import corpora , models
from limpiador import Limpiador
from postagger import SpanishParser, traductor
from supervisado.Datos import Datos
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from collections import defaultdict
from kimchi import load_vocab_tokens2
from itertools import imap,ifilter
from neural_hate_classifier import neural_classifier
from sklearn.preprocessing import RobustScaler







class Hate_classifier:

    paser = None
    limpiador = None
    dictionary = None
    clf = None
    vocab_features = None
    hate_bigrams = None
    encAtributos = None
    vocab_dict = None
    pre_filter = True
    using_bigrams = False
    normalizador = None
    gensim_dict = None
    clasi_name = "Default"
    tdfidf = None


    def __init__(self,pre_filter=True,using_bigrams=False,n_categorical_values=2):
        self.parser = SpanishParser(morphology_path="morfo/es-morphology2.txt",context_path="morfo/es-context2.txt",lexicon_path= "morfo/es-lexicon4.txt")
        self.limpiador =  Limpiador(None)
        aux_dic = corpora.Dictionary()
        aux_dic = aux_dic.load('program_data/tweets.dict')
        self.gensim_dict = aux_dic
        self.pre_filter = pre_filter
        self.using_bigrams = using_bigrams

        if pre_filter:
            self.vocab_dict ,  num_category, filenames = load_vocab_tokens2(aux_dic,False)

        self.dictionary = defaultdict(lambda:-1)

        if self.using_bigrams:
            self.hate_bigrams,self.vocab_features = self.load_filtered_bigram_features()
        else:
            self.vocab_features = self.load_filtered_features()
        self.encAtributos = preprocessing.OneHotEncoder(n_values=n_categorical_values,sparse=False)
        for key in aux_dic.token2id.keys():
            self.dictionary[key] = aux_dic.token2id[key]

        self.clf = None



    def fit(self,norm=False,clf_name = "neural"):
        datos = Datos(r"training_set/hate_training_set.data",True)

        dd = datos.datos[:,:-1]
        if norm:
            self.tdfidf = models.TfidfModel()
            self.tdfidf = self.tdfidf.load('program_data/hate.tfidf_model')
            self.normalizador = RobustScaler()
            self.normalizador.fit(dd)
            X = self.normalizador.transform(dd)
        else:
            X = self.encAtributos.fit_transform(dd)

        Y = datos.datos[:,-1]

        self.clasi_name = clf_name

        if clf_name == "bayes":
            self.clf = MultinomialNB(fit_prior=True,alpha=2.1e-06)
        elif clf_name == "neural":
            self.clf = neural_classifier(X.shape[1],X.shape[0])
        else:
            print("Wrong classifier, choose another one!")

        self.clf.fit(X, Y)
        print("Fit finished!")


    def predict(self,tweet):
        text = self.limpiador.clean(tweet)
        cadena = self.parser.parse(text,tokenize=True)
        if(len(cadena) < 1):
            return [0]
        cadenas = cadena.split()[0]
        plain_token = []

        vector = []
        for term in cadenas:
            meta = traductor(term[1])
            text = term[0]
            if meta in ["Adjetivo","Verbo","Nombre"]:
                if self.dictionary[text] != -1:
                    vector.append(self.dictionary[text])
                    if  self.clasi_name == "neural":
                        plain_token.append(text)


        if self.pre_advanced_filter(vector[:]):
            if self.clasi_name != "neural":
                binary_vector= map(lambda term_vocab: 1 if term_vocab in vector else 0,self.vocab_features)
                if self.using_bigrams:
                    tweet = self.tweet2digram(vector)
                    binary_vector = binary_vector + map(lambda term_vocab: 1 if term_vocab in tweet else 0, self.hate_bigrams) 

                binary_vector = np.array(binary_vector)
                binary_vector = binary_vector.reshape(1,-1)
                X = self.encAtributos.transform(binary_vector)
            else:
                bow = self.gensim_dict.doc2bow(plain_token)
                tfidf_tuples = self.tdfidf[bow]
                id_tfidf_dict = dict(tfidf_tuples)
                float_vector= map(lambda term_vocab: id_tfidf_dict[term_vocab] if term_vocab in vector else 0.0,self.vocab_features)
                float_vector = np.array(float_vector)
                float_vector = float_vector.reshape(1,-1)
                X = self.normalizador.transform(float_vector)



            if self.clf.predict(X)[0] == 1:
                return [1]
            else:
                return[2]
        return [0]
    



    ## Carga los atributos tanto de bigramas de odio como los unigramas
    def load_filtered_bigram_features(self):
        dictionary = corpora.Dictionary()
        dictionary = dictionary.load('program_data/tweets.dict')
        f = open("training_set/filtered_features.txt","r")
        line = f.readline()
        frags = line.split(",")
        frags = frags[:-1]
        bigram_terms = []
        simple_terms = []
        for term in frags:
            if "-" in term:
                tupla = ()
                subterminos = term.split("-")
                for subterm in subterminos:
                    tupla = tupla + (dictionary.token2id[subterm], )
                bigram_terms.append(tupla)
            else:
                simple_terms.append(dictionary.token2id[term])

        return bigram_terms,simple_terms


        
       
        
    ## Cargar los atributos previamente obtenidos
    def load_filtered_features(self):
        dictionary = corpora.Dictionary()
        dictionary = dictionary.load('program_data/tweets.dict')
        f = open("training_set/filtered_features.txt","r")
        line = f.readline()
        frags = line.split(",")
        frags = frags[:-1]
        result = [ dictionary.token2id[term] for term in frags]
        f.close()
        return result


    def pre_advanced_filter(self,tweet):
        if not self.pre_filter:
            return 1
        
        final_tweet = list(ifilter(lambda x: x is not None,imap(lambda term: self.vocab_dict[term], tweet)))


        if len(final_tweet) == 0:
            return 0

        
        #final_list = imap(lambda corp: list(ifilter(lambda x: x != -99,imap(lambda x: x[0] if x[1] == -1 else vocab_dict[x[0]],corp))), corpus)

        #Si nuestro tweet contiene una palabra de maldad entonces convertimos todas
        # las negativas (malas leves) a positivas (verdaderas malas) y quitamos los 0 (palabra de maldad)
        #  en otro caso quitamos los ceros y las que son negativa

        if 0 in final_tweet:
            final_hate_inside = list(filter(lambda x: x!=0,imap(abs,final_tweet)))
        else:
            final_hate_inside = list(filter(lambda x: x > 0,final_tweet))


        return len(final_hate_inside) > 0


    """
    Transforma un tweet a digramas
    """
    def tweet2digram(self,tweet):
        digram_tweet = set()
        for i in range(len(tweet)-1):
            digram_tweet.add((tweet[i],tweet[i+1]))

        return list(digram_tweet)


"""
if __name__ == "__main__":
    hcl = Hate_classifier()
    hcl.fit()
    hcl.predict("ayer estuve en el orgullo gay con mi amigo sarasa")
"""




