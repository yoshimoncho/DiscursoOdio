from gensim import corpora,models, similarities
from collections import defaultdict
from itertools import ifilter, imap
from six import iteritems
import pickle
import numpy as np
import codecs






def main(charged_filtered =False,add_more=True, tf_idf_used=False,freq_used=False,create_bigrams_strucs=False,add_bigrams = False,charged_filtered_bigrams=False):
	if tf_idf_used:
		freq_used = False
		add_bigrams = False
		charged_filtered_bigrams= False
		corpus = corpora.MmCorpus('program_data/tweets_corpus.mm')
		num_docs = corpus.num_docs
		num_terms = corpus.num_terms
		corpus = transform_tf_idf_corpus(corpus)
	else:
		corpus = corpora.MmCorpus('program_data/tweets_corpus.mm')
		num_docs = corpus.num_docs
		num_terms = corpus.num_terms

	if(create_bigrams_strucs):
		generate_bigrams_struct()

	
	if(add_bigrams):
		bigram_corpus = pickle.load( open( "training_set/bigrams_struct.p", "rb" ) )
		vocab_features = pickle.load( open( "training_set/vocab_features.p", "rb" ) )
		if not charged_filtered_bigrams:
			hate_bigrams = create_hate_bigrams_dict(vocab_features,bigram_corpus)
		print("Bigram Corpus, loaded...")



	corpus = (list(corpus))
	print("Corpus, loaded...")

	labels_dict , traning_tweets_ids= extract_labels("training_set/tagged_3000.txt")
	aux_ids = traning_tweets_ids[:]
	traning_tweets_ids.reverse()

	print("Training tweets ids, loaded...")
	print("Number training elements: "+ str(len(traning_tweets_ids)))

	diccionario_id2word = pickle.load( open( "program_data/dic_id2word.p", "rb" ) )

	

	#Estraer tweets de entrenamiento del corpus por id O(|Tweets hasta encontrar todos training ids|)
	i = 0
	training_tweets = []
	freq_dicts = []
	tfidf_dicts = []
	total_token_tweets = []

	if add_bigrams:
		training_bigram_tweets = []

	while len(traning_tweets_ids) > 0:
		if(traning_tweets_ids[-1]== abs(corpus[i][0][0])):
			traning_tweets_ids.pop()
			list_token_tweet = map(lambda par: par[0], corpus[i][1:])
			total_token_tweets += list_token_tweet
			training_tweets.append(list_token_tweet)
			if add_bigrams:
				training_bigram_tweets.append(tweet2digram(bigram_corpus[i][1:]))
			if tf_idf_used:
				tfidf_dicts.append(dict(corpus[i][1:]))
			if freq_used:
				freq_dicts.append(dict(corpus[i][1:]))



		i+=1


	#Carga de atributos

	vocab_features = []
	if(charged_filtered_bigrams):
		hate_bigrams,vocab_features = load_filtered_bigram_features()
	elif(charged_filtered):
		vocab_features = load_filtered_features()
	else:
		if not add_bigrams:
			vocab_features = pickle.load( open( "training_set/vocab_features.p", "rb" ) )
		#Extraemos atributos adicionales (terminos del corpus ordenados por frecuencia)
		if add_more:
			vocab_features = list(set(total_token_tweets))

	print("Vocab features, loaded...")
	if not add_bigrams:
		print("Number of features: "+ str(len(vocab_features)))
	else:
		print("Number of features: "+ str(len(vocab_features)+len(hate_bigrams)))



	#Matriz a rellenar en los tweets con el i-esimo atributo
	#Hay dos opciones: modelo esta o no (1-0) o poner la frecuencia
	#features_matrix = np.zeros((len(vocab_features),len(training_tweets)))

	if tf_idf_used:
		final_matrix = map(lambda tweet,tfidf_dict: map(lambda term_vocab: tfidf_dict[term_vocab] if term_vocab in tweet else 0.0,vocab_features),training_tweets,tfidf_dicts)
	elif freq_used:
		final_matrix = map(lambda tweet,freq_dic: map(lambda term_vocab: int(freq_dic[term_vocab]) if term_vocab in tweet else 0,vocab_features),training_tweets,freq_dicts)
	else:
		final_matrix = map(lambda tweet: map(lambda term_vocab: 1 if term_vocab in tweet else 0,vocab_features),training_tweets)

	if add_bigrams:
		final_bigram_matrix = map(lambda tweet: map(lambda term_vocab: 1 if term_vocab in tweet else 0, hate_bigrams)   ,training_bigram_tweets)



	#Creacion del archivo del conjunto de entrenamiento
	f = open("training_set/hate_training_set.data","w")
	#Numero de elementos en la muestra
	f.write(str(len(training_tweets))+ "\n")

	#Atributos a usar (en nuestro caso son terminos con valores 0 si no estan en el tweet y 1 si lo estan | frecuecia)
	for i in range(len(vocab_features)):
		f.write(diccionario_id2word[vocab_features[i]]+",")

	if add_bigrams:
		for i in range(len(hate_bigrams)):
			f.write(diccionario_id2word[hate_bigrams[i][0]]+"-"+diccionario_id2word[hate_bigrams[i][1]]+",")
	f.write("Clase\n")

	if add_bigrams:
		num_atributos = len(vocab_features) + len(hate_bigrams)
	else:
		num_atributos = len(vocab_features)

	#Tipo de atributos (por ahora todos discretos 0 o 1)
	for i in range(num_atributos):
		if freq_used or tf_idf_used:
			f.write("Continuo,")
		else:
			f.write("Nominal,")
	#Etiqueta de la clase
	f.write("Nominal\n")

	#Valor de atributos
	counter = 0

	if add_bigrams:
		final_matrix = map(lambda x,y: x+y, final_matrix,final_bigram_matrix)
	
	for valores_tweets in final_matrix:
		for i in range(len(valores_tweets)):
			f.write(str(valores_tweets[i])+",")
		#Etiqueta de la clase
		f.write(str(labels_dict[aux_ids[counter]])+"\n")
		counter+= 1
	f.close()


## Carga los atributos tanto de bigramas de odio como los unigramas
def load_filtered_bigram_features():
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
def load_filtered_features():
	dictionary = corpora.Dictionary()
	dictionary = dictionary.load('program_data/tweets.dict')
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
	dictionary = dictionary.load('program_data/tweets.dict')
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
	f = codecs.open(filename,"r", "utf-8") 
	#open(filename,"r")
	#Diccionario de id:tweet => etiqueta
	labels_dict = {}
	ids_list = []
	n_id = 0
	for line in f:
		frags = line.split(";||;")
		if len(frags) != 3:
			print("Wrong format, maybe some labels are lost!")
			exit(-1)
		labels_dict[int(frags[0][3:])] = int(frags[2])
		ids_list.append(int(frags[0][3:]))

	print("Number of labels: "+ str(len(labels_dict)))
	
	labels, counts = np.unique(labels_dict.values(),return_counts=True)
	for i  in range(len(labels)):
		print("Label: "+ str(labels[i])+" Freq: "+str(counts[i]))

	f.close()
	return labels_dict,ids_list





"""
Genera un corpus que permita dar importancia
a la situacion de cada termino dentro del tweet
anteriormente: (id,freq)
"""

def generate_bigrams_struct(numero=10):
	dictionary = corpora.Dictionary()
	dictionary = dictionary.load('program_data/tweets.dict')
	texts = []
	perdidos = 0
	lista_ids = []
	for j in range(1,numero+1):

		file = open("treatcorpus/treat_basegrande"+str(j)+"_clean.txt","r")
		contador = 0 
		for line in file:
			if line != '\n':
				fragmentos = line.split(";||;")
				linea = fragmentos[1]
				identificador = int(fragmentos[0])
				if linea != "\n":
					if linea[-1] != "\n":
						print("Ojo al corte")
					tweet = linea[0:-1].split()
					final_tweet = [identificador]+ map(lambda term: dictionary.token2id[term],tweet)
					texts.append(final_tweet)
				else:
					perdidos += 1
		file.close()

	pickle.dump(texts,open( "training_set/bigrams_struct.p", "wb" ))

"""
Genera bigrama de terminos asociados al odio
"""
def create_hate_bigrams_dict(vocab_features,bigram_corpus):
	vocab = defaultdict(lambda:False)
	bigram_set = set()

	for word in vocab_features:
		vocab[word] = True


	for tweet_full in bigram_corpus:
		#Quitamos el id de los tweets para hacer los bigramas
		tweet = tweet_full[1:]
		for i in range(len(tweet)-1):
			if(i > 0):
				if(vocab[tweet[i]]):
					bigram_set.add((tweet[i-1],tweet[i]))

			if(vocab[tweet[i]]):
				bigram_set.add((tweet[i],tweet[i+1]))
	return list(bigram_set)


"""
Transforma un tweet a digramas
"""
def tweet2digram(tweet):
	digram_tweet = set()
	for i in range(len(tweet)-1):
		digram_tweet.add((tweet[i],tweet[i+1]))

	return list(digram_tweet)



def transform_tf_idf_corpus(corpus):
	tf_idf_corpus = []

	for tweet in corpus:
		tf_idf_corpus.append(tweet[1:])

	tfidf = models.TfidfModel(tf_idf_corpus)
	corpus_treated = tfidf[tf_idf_corpus]
	print("guardado...")
	tfidf.save('program_data/hate.tfidf_model')
	dictionary = corpora.Dictionary()
	dictionary = dictionary.load('program_data/tweets.dict')

	return [ corpus[i][:1]+corpus_treated[i] for i in range(len(corpus))]









if __name__ == "__main__":
	print("Ejecutando training_set_generator.py ...")
	#vocab_features = pickle.load( open( "training_set/vocab_features.p", "rb" ) )
	#lista = create_hate_bigrams_dict(vocab_features,create_strucs = False)
	#print(len(lista))
	#print(top_term_freq(cuttoff= 10,most_common=True))
	#main(True,add_more=True,freq_used=False) primero usado
	#main(charged_filtered =True,add_more=False, freq_used=False,create_bigrams_strucs=False,add_bigrams = True,charged_filtered_bigrams=False)

	main(charged_filtered =True,add_more=False, tf_idf_used=True,freq_used=False,create_bigrams_strucs=False,add_bigrams = False,charged_filtered_bigrams=False)
