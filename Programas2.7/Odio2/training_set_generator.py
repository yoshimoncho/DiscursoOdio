from gensim import corpora
from collections import defaultdict
from itertools import ifilter, imap
from six import iteritems
import pickle
import numpy as np






def main(charged_filtered =False):
	corpus = corpora.MmCorpus('tweets_corpus.mm')
	num_docs = corpus.num_docs
	num_terms = corpus.num_terms
	corpus = (list(corpus))
	print("Corpus, loaded...")
	vocab_features = []
	if(charged_filtered):
		vocab_features = load_filtered_features()
	else:
		vocab_features = pickle.load( open( "training_set/vocab_features.p", "rb" ) )
		#Extraemos atributos adicionales (terminos del corpus ordenados por frecuencia)
		new_features = top_term_freq(cuttoff = 100000,most_common=True)
		vocab_features = vocab_features + new_features

	print("Vocab features, loaded...")
	print("Number of features: "+ str(len(vocab_features)))
	traning_tweets_ids = pickle.load( open( "training_set/training_tweets.p", "rb" ) )
	traning_tweets_ids = traning_tweets_ids
	aux_ids = traning_tweets_ids[:]
	traning_tweets_ids.reverse()

	print("Training tweets ids, loaded...")
	print("Number training elements: "+ str(len(traning_tweets_ids)))

	diccionario_id2word = pickle.load( open( "dic_id2word.p", "rb" ) )

	labels_dict = extract_labels("training_set/tagged_1000.txt")

	#Estraer tweets de entrenamiento del corpus por id O(|Tweets hasta encontrar todos training ids|)
	i = 0
	training_tweets = []

	while len(traning_tweets_ids) > 0:
		if(traning_tweets_ids[-1]== abs(corpus[i][0][0])):
			traning_tweets_ids.pop()
			training_tweets.append(list(map(lambda par: par[0],corpus[i][1:])))
		i+=1

	#Matriz a rellenar en los tweets con el i-esimo atributo
	#features_matrix = np.zeros((len(vocab_features),len(training_tweets)))

	final_matrix = map(lambda tweet: map(lambda term_vocab: 1 if term_vocab in tweet else 0,vocab_features),training_tweets)
	#Creacion del archivo del conjunto de entrenamiento
	f = open("training_set/hate_training_set.data","w")
	#Numero de elementos en la muestra
	f.write(str(len(training_tweets))+ "\n")

	#Atributos a usar (en nuestro caso son terminos con valores 0 si no estan en el tweet y 1 si lo estan)
	for i in range(len(vocab_features)):
		f.write(diccionario_id2word[vocab_features[i]]+",")
	f.write("Clase\n")

	#Tipo de atributos (por ahora todos discretos 0 o 1)
	for i in range(len(vocab_features)):
		f.write("Nominal,")
	#Etiqueta de la clase
	f.write("Nominal\n")

	#Valor de atributos
	counter = 0
	
	for valores_tweets in final_matrix:
		for i in range(len(valores_tweets)):
			f.write(str(valores_tweets[i])+",")
		#Etiqueta de la clase
		f.write(str(labels_dict[aux_ids[counter]])+"\n")
		counter+= 1
	f.close()



## Cargar los atributos previamente obtenidos
def load_filtered_features():
	dictionary = corpora.Dictionary()
	dictionary = dictionary.load('tweets.dict')
	f = open("training_set/filtered_features.txt","r")
	line = f.readline()
	frags = line.split(",")
	frags = frags[:-1]
	result = [ dictionary.token2id[term] for term in frags]
	f.close()
	return result
#Top palabras mas/menos frecuentes en los tweets (acotado por el numero de tweets)
def top_term_freq(cuttoff = 10,most_common=True):
	dictionary = corpora.Dictionary()
	dictionary = dictionary.load('tweets.dict')
	#diccionario_id2word = pickle.load( open( "dic_id2word.p", "rb" ) )
	print("Dictionary, loaded...")
	print("Dictionary size: "+str(len(dictionary.dfs.keys())))
	lista = [ (v,k) for k,v in iteritems(dictionary.dfs)]
	lista.sort(reverse=most_common)
	top = lista[:cuttoff]
	#salida = map(lambda tupla: (tupla[0], diccionario_id2word[tupla[1]]),top)
	salida = map(lambda tupla: tupla[1],top)
	return salida


#Los tweet_ids deberan de tener el mismo orden que el archivo desde el que se lee (por eficiencia)
def extract_labels(filename):
	f = open(filename,"r")
	#Diccionario de id:tweet => etiqueta
	labels_dict = {}
	n_id = 0
	for line in f:
		frags = line.split(";||;")
		if len(frags) != 3:
			print("Wrong format, maybe some labels are lost!")
			exit(-1)
		labels_dict[int(frags[0][3:])] = int(frags[2])

	print("Number of labels: "+ str(len(labels_dict)))
	
	labels, counts = np.unique(labels_dict.values(),return_counts=True)
	for i  in range(len(labels)):
		print("Label: "+ str(labels[i])+" Freq: "+str(counts[i]))

	f.close()
	return labels_dict














if __name__ == "__main__":
	#print(top_term_freq(cuttoff= 10,most_common=True))
	main(True)
