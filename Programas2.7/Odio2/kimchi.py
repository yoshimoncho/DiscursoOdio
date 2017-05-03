from gensim import corpora,matutils
from math import log
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import ifilter, imap
import pickle
from six import iteritems
import numpy as np
import codecs


# 
# Atributos de corpus
# self.num_docs, self.num_terms


def main():
	#save_tweet_tokens()
	#heaps_law()
	
	
	dictionary = corpora.Dictionary()

	
	corpus = corpora.MmCorpus('program_data/tweets_corpus.mm')
	num_docs = corpus.num_docs
	num_terms = corpus.num_terms
	corpus = (list(corpus))
	print("Corpus, loaded...")

	dictionary = dictionary.load('program_data/tweets.dict')
	print("Dictionary, loaded...")
	
	pre_tagger_corpus, vocab_dict,num_category,dict_etiquetas = category_tagger_advanced(corpus,dictionary)
	print("Clasificados:" + str(len(pre_tagger_corpus)))


	term_dic = defaultdict(lambda: -1)
	

	#Creamos un diccionario de terminos dict[clave] -> diccionario de categorias
	#El diccionario de categorias[num categoria] -> frec del termino para esa categoria
	# O(N) donde N es el numero total de terminos

	"""
	#MINIPRUEBA
	print("Documentos clasificados: " + str(len(pre_tagger_corpus)))
	puta = 0
	for doc in pre_tagger_corpus:
		i = 0
		for termino in doc:
			if i == 0:
				i+=1
			else:
				if termino[0] == dictionary.token2id["puta"]:
					puta+= termino[1]
	print("Veces puta: "+ str(puta))
	"""

	contador = 0
	for doc in pre_tagger_corpus:
		i = 0
		
		for termino in doc:
			if i == 0:
				#Puede haber mas de una categoria para un tweet
				categorias = termino[1:]
				if(len(categorias) > 1):
					print("Tweet con mas de una categorias: "+ str(abs(termino[0])))
					contador+=1
				i = 1
			else:
				for categoria in categorias:
					if term_dic[termino[0]] == -1:
						term_dic[termino[0]] = defaultdict(lambda: -1)
						term_dic[termino[0]][categoria] = termino[1]
					else:
						if term_dic[termino[0]][categoria] == -1:
							term_dic[termino[0]][categoria] = termino[1]
						else:
							term_dic[termino[0]][categoria] += termino[1]

	print("Totales con mas de 1 categoria "+str(contador))
	#Calculamos los pesos de cada documento para su categoria/as

	pesos_categorias = defaultdict(lambda: [])

	#Guardamos  los pesos maximos de cada tweet para una de sus categorias
	#Permite obtener las palabras mas importantes de un tweet


	doc_max_pesos = defaultdict(lambda: [])
	#Longitud minima de un tweet
	min_lenght = 4
	for doc in pre_tagger_corpus:
		if len(doc) < min_lenght+1:
			continue
		pesos = defaultdict(lambda: 0)
		max_pesos = defaultdict(lambda: (-1,-1))
		i = 0
		for termino in doc:
			if i == 0:
				categorias = termino[1:]
				id_tweet = abs(termino[0])
				i = 1
			else:
				for categoria in categorias:
					peso_aux = tf(termino[0], categoria, term_dic) * icf(termino[0],num_category, term_dic)
					max_pesos[categoria] = max((peso_aux,termino[0]),max_pesos[categoria])
					pesos[categoria] += peso_aux*termino[1]
		pesos[categoria] = pesos[categoria]/len(doc)

		for categoria in categorias:
			doc_max_pesos[id_tweet].append(max_pesos[categoria][1])
			pesos_categorias[categoria].append((pesos[categoria],id_tweet))

	procentaje = 1
	claves = pesos_categorias.keys()
	print("Num claves:" + str(len(claves)))
	acumulador = 0
	for clave in claves:
		ordenados = sorted(pesos_categorias[clave],reverse=True)
		indice = int(len(ordenados) * procentaje)
		if indice < 1:
			pesos_categorias[clave] = ordenados[:-1]
		else:
			pesos_categorias[clave] = ordenados[:indice]


	#id2word = load_dic_id2word(dictionary)

	#print(doc_max_pesos[pesos_categorias[1][0][1]])
	#id2docid_posrel = load_id2docid_posrel()
	#print(map(lambda x: (x[0],id2docid_posrel[x[1]] ) ,pesos_categorias[3]))
	tagger_output(pesos_categorias,dict_etiquetas)



# Calcula el icf de un termino, el icf es independiente de la categoria
def icf(term,num_category, dictionary):
	return log(num_category) - log(len(dictionary[term]))

#Calcula la frecuencia de un termino en una categoria
def tf(term, category, dictionary):
	return dictionary[term][category]


def category_tagger_basic(corpus,dictionary):
	vocab_dict , num_category, filenames = load_vocab_tokens(dictionary,False)
	print("Categoy words, loaded...")
	

	# Necesario volver hacia atras.... saber el tweet
	# Prefiltrado, considerando que los contenedores de bag words son de tal categoria
	# Unos 4s de ejecucion (bastante eficiente)
	
	
	final_list = imap(lambda corp: list(ifilter(lambda x: x is not None,imap(lambda x: x[0] if x[1] == -1 else vocab_dict[x[0]],corp))), corpus)
	resultado_sinfiltrar = imap(lambda x,y: [(x[0],)+tuple(set(x[1:]))]+y[1:] if len(x) > 1 else None,final_list,corpus)

	#Obtenemos el corpus pre_clasificado, faltaria sacar el ranking de pesos.
	
	pre_tagger_corpus = list(filter(lambda x: x is not None, resultado_sinfiltrar))
	return pre_tagger_corpus,vocab_dict,num_category, filenames


#Tagger de categorias con soporte de palabras de odio
def category_tagger_advanced(corpus,dictionary):
	vocab_dict ,  num_category, filenames = load_vocab_tokens2(dictionary,False)
	print("Categoy words, loaded...")


	# [id, clase1,clase2]
	
	final_list = imap(lambda corp: list(ifilter(lambda x: x is not None,imap(lambda x: x[0] if x[1] == -1 else vocab_dict[x[0]],corp))), corpus)
	#Si nuestro tweet contiene una palabra de maldad entonces convertimos todas
	# las negativas (malas leves) a positivas (verdaderas malas) y quitamos los 0 (palabra de maldad)
	#  en otro caso quitamos los ceros y las que son negativa


	final_hate_inside  = imap(lambda elems: [elems[0]]+ list(filter(lambda x: x!= 0,imap(abs,elems[1:]))) if 0 in elems else [elems[0]]+ list(filter(lambda x: x>0,elems[1:]))  ,final_list)

	resultado_sinfiltrar = imap(lambda x,y: [(x[0],)+tuple(set(x[1:]))]+y[1:] if len(x) > 1 else None,final_hate_inside,corpus)

	#Obtenemos el corpus pre_clasificado, faltaria sacar el ranking de pesos.
	
	pre_tagger_corpus = list(filter(lambda x: x is not None, resultado_sinfiltrar))
	return pre_tagger_corpus, vocab_dict,num_category, filenames

#Carga los terminos del vocabulario con su correspondiente clase (primer modelo)
def load_vocab_tokens(dictionary,stemming=False,foldername = "vocabulario",filenames=["discapacidad.txt","genero.txt","raza.txt","politica.txt"]):
	vocabulario = defaultdict(lambda:None)
	dict_etiquetas = defaultdict(lambda:"Empty")
	i = 0
	for file in filenames:
		print(foldername+"/"+file)
		f = open(foldername+"/"+file,"r")
		dict_etiquetas[i] = file
		for line in f:
			token_vocab = line[:-1]
			try:
				vocabulario[dictionary.token2id[token_vocab]] = i
			except KeyError, e:
				print("Didn't find: '"+token_vocab+"' in tweets" )
		f.close()
		i += 1

	#Guardar vocabulario para generacion de atributos (palabras de odio)
	pickle.dump(list(vocabulario.keys()),open( "training_set/vocab_features.p", "wb" ))
	return vocabulario,i+1,dict_etiquetas

#Carga los terminos del vocabulario con su correspondiente clase (segundo modelo)
def load_vocab_tokens2(dictionary,stemming=False,foldername = "vocabulario2",filenames=["discapacidad.txt","genero.txt","raza.txt","politica.txt","religion.txt","clases.txt"],hate="maldad.txt"):
	vocabulario = defaultdict(lambda:None)
	dict_etiquetas = defaultdict(lambda:"Empty")
	i = 1
	for file in filenames:
		print(foldername+"/"+file)
		f = open(foldername+"/"+file,"r")
		dict_etiquetas[i] = file
		for line in f:
			frags_vocab = line.split()
			token_vocab = frags_vocab[0]
			token_hate = frags_vocab[1]
			try:
				if token_hate == "-2":
					vocabulario[dictionary.token2id[token_vocab]] = -i
				else:
					vocabulario[dictionary.token2id[token_vocab]] = i

			except KeyError, e:
				print("Didn't find: '"+token_vocab+"' in tweets" )
		f.close()
		i += 1
	f = open(foldername+"/"+hate,"r")

	for line in f:
		token_vocab = line[:-1]
		try:
			vocabulario[dictionary.token2id[token_vocab]] = 0
		except KeyError, e:
			print("Didn't find: '"+token_vocab+"' from hate doc" )
	f.close()

	#Guardar vocabulario para generacion de atributos
	pickle.dump(list(vocabulario.keys()),open( "training_set/vocab_features.p", "wb" ))
	return vocabulario,i,dict_etiquetas



def tagger_output(pesos_categorias,dict_etiquetas):
	
	#Permite observar el resultado de la salida (se creo al crear el corpus)
	id2docid_posrel = load_id2docid_posrel()
	posiciones_relativas = []
	f = codecs.open("tagger_output/new_output.txt","w", "utf-8")

	for categoria in pesos_categorias.keys():
		posiciones_relativas = posiciones_relativas +  map(lambda x: id2docid_posrel[x[1]]+(categoria,),pesos_categorias[categoria])


	posiciones_relativas = sorted(posiciones_relativas)
	pos_aux = -1
	doc_tweet_anterior = -1
	doc_tweet_pos = -1
	id_seleccionados = []
	while len(posiciones_relativas) > 0:
		posicion = posiciones_relativas.pop(0)
		if doc_tweet_anterior == posicion[0] and doc_tweet_pos == posicion[1]:
			continue
		doc_tweet_anterior = posicion[0]
		doc_tweet_pos = posicion[1]
		if posicion[0] != pos_aux:
			if pos_aux != -1:
				aux_file.close()
			pos_aux = posicion[0]
			aux_file = codecs.open("rawcorpus/basegrande"+str(pos_aux)+".txt", "r", "utf-8")
		while True:
			linea = aux_file.readline()
			if linea is not None:
				frags = linea.split(";||;")
				if frags[0] == "n="+str(posicion[1]):
					id_seleccionados.append(int(frags[1][3:]))
					f.write(frags[1]+";||;"+frags[4]+";||;\n")
					break
	#Almacenar la lista de tweets seleccionados para generar el conjunto de entrenamiento
	pickle.dump(id_seleccionados,open( "training_set/training_tweets.p", "wb" ))
	f.close()
	aux_file.close()

		
"""
Imprimer una lista de tweets clasificados:
Cada elemento de la lista es una lista con primer elemento id del tweet y el resto son categorias
"""
def print_lista(final_list,id2docid_posrel):
	for elemento in final_list:
		print("Coordenadas: "+ str(id2docid_posrel[abs(elemento[0])])+" Clase: "+str(elemento[1]) )


"""
Devuelve un diccionario guardado clave: id del tweet y valor: (doc_id, pos relativa en el documento (n))
"""
def load_id2docid_posrel():
	print("TweetId2Docid_posrel, loaded...")
	return pickle.load( open( "program_data/id2docid_posrel.p", "rb" ) )	



"""
Muestra una grafica asociada a la ley de heap cuyos datos se almacenan en heap.txt
"""
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


"""
Crea apartir de un corpora.Dictionary
un nuevo diccionario de python con clave: id y valor:string del token
Aconsejable guardar este diccionario para posteriores usos.
"""

def create_dic_id2word(dictionary):
	return dict([(v,k) for k,v in iteritems(dictionary.token2id)])


"""
Carga un diccionario de python dic_id2word con clave: id y valor:string del token
"""

def load_dic_id2word(dictionary):
	return pickle.load( open( "program_data/dic_id2word.p", "rb" ) )

#Numero indica el numero de archivos a leer de treatcorpus
def save_tweet_tokens(numero=10):

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
					texts.append(linea[0:-1].split())
					lista_ids.append(identificador)
				else:
					perdidos += 1
					#print("Doc: "+ str(j) +" Num_Tweet: "+str(identificador))
		file.close()

	print("Lost tweets on create corpora: " + str(perdidos))
	dictionary = corpora.Dictionary(texts)

	dic_id2word = create_dic_id2word(dictionary)
	pickle.dump(dic_id2word,open( "program_data/dic_id2word.p", "wb" ))

	#corpus = [dictionary.doc2bow(text) for text in texts]
	corpus = map(lambda texto,numero: [(-numero,-1)] + dictionary.doc2bow(texto) ,texts,lista_ids)
	corpora.MmCorpus.serialize('program_data/tweets_corpus.mm',corpus)
	dictionary.save('program_data/tweets.dict')






if __name__ == "__main__":
	print("Ejecutando kimchi.py ...")
	main()
	#save_tweet_tokens()