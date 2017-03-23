from gensim import corpora,matutils
from math import log
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import ifilter, imap
import pickle


# 
# Atributos de corpus
# self.num_docs, self.num_terms


def main():
	#save_tweet_tokens()
	#heaps_law()
	
	
	dictionary = corpora.Dictionary()
	corpus = corpora.MmCorpus('tweets_corpus.mm')
	num_docs = corpus.num_docs
	num_terms = corpus.num_terms
	corpus = list(corpus)
	print("Corpus, loaded...")
	dictionary = dictionary.load('tweets.dict')
	print("Dictionary, loaded...")
	

	#new_vec = dictionary.doc2bow("kpop".split())
	#print(list(corpus))
	#print(dictionary)
	#print(len(dictionary))
	#print(corpus.num_terms,corpus.num_docs)
	
	
	vocab_dict , num_clases = load_vocab_tokens(dictionary,False)
	print("Categoy words, loaded...")
	pre_tagger = []

	# Necesario volver hacia atras.... saber el tweet
	# Prefiltrado, considerando que los contenedores de bag words son de tal categoria
	# Unos 4s de ejecucion (bastante eficiente)
	
	
	final_list = list(ifilter(lambda z: len(z) > 1,imap(lambda corp: list(ifilter(lambda x: x != -1,imap(lambda x: x[0] if x[1] == -1 else vocab_dict[x[0]],corp))), corpus)))
	




def load_vocab_tokens(dictionary,stemming=False,foldername = "vocabulario",filenames=["discapacidad.txt","genero.txt","raza.txt"]):
	vocabulario = defaultdict(lambda:-1)
	
	i = 0
	for file in filenames:
		print(foldername+"/"+file)
		f = open(foldername+"/"+file,"r")
		for line in f:
			token_vocab = line[:-1]
			try:
				vocabulario[dictionary.token2id[token_vocab]] = i
			except KeyError, e:
				print("Didn't find: '"+token_vocab+"' in tweets" )
		i += 1
		
	return vocabulario,i+1;
		
"""
Imprimer una lista de tweets clasificados:
Cada elemento de la lista es una lista con primer elemento id del tweet y el resto son categorias
"""
def print_lista(final_list,id2docid_posrel):
	for elemento in final_list:
		if elemento[1] == 2:
			print("Coordenadas: "+ str(id2docid_posrel[abs(elemento[0])])+" Clase: "+str(elemento[1]) )


"""
Devuelve un diccionario guardado clave: id del tweet y valor: (doc_id, pos relativa en el documento (n))
"""
def load_id2docid_posrel():
	print("TweetId2Docid_posrel, loaded...")
	return pickle.load( open( "id2docid_posrel.p", "rb" ) )	


def heaps_law():
	file = open("heap.txt","r")
	x_axis = []
	y_axis = []
	for line in file:
		frags = line.split()
		x = int(frags[0])
		x = log(x)
		x_axis.append(x)
		y = int(frags[1])
		y = log(y)
		y_axis.append(y)
	plt.plot(x_axis, y_axis)
	plt.show()



def save_tweet_tokens(numero=11):

	texts = []
	perdidos = 0
	lista_ids = []
	for j in range(1,numero):

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
					texts.append(linea[0:-1].split())
					lista_ids.append(identificador)
				else:
					perdidos += 1
					#print("Doc: "+ str(j) +" Num_Tweet: "+str(identificador))

	print("Lost tweets on create corpora: " + str(perdidos))
	dictionary = corpora.Dictionary(texts)
	#corpus = [dictionary.doc2bow(text) for text in texts]
	corpus = map(lambda texto,numero: [(-numero,-1)] + dictionary.doc2bow(texto) ,texts,lista_ids)
	corpora.MmCorpus.serialize('tweets_corpus.mm',corpus)
	dictionary.save('tweets.dict')






if __name__ == "__main__":
	main()