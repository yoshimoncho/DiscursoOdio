import numpy as np
from collections import defaultdict
from gensim import corpora, models, similarities
import pickle


class Cosine_Hate_Classifier:

	list_labels = []
	index = None
	tfidf = None
	k_neighbours = 0
	num_terms = 0
	vectors_train = []
	counter_class = 1
	dict_id2word = None

	def __init__(self,corpus_tweets_file,k_neighbours=3):
		self.k_neighbours = k_neighbours
		corpus = corpora.MmCorpus(corpus_tweets_file)
		self.index = None
		self.list_labels = []
		self.tfidf = models.TfidfModel(corpus)
		self.num_terms = corpus.num_terms
		self.vectors_train= []
		self.dict_id2word = pickle.load(open( "program_data/dic_id2word.p", "rb" ))
		



	def fit(self,vectors_train,y_train):
		self.index = similarities.SparseMatrixSimilarity(self.tfidf[vectors_train],num_features=self.num_terms)
		self.vectors_train = vectors_train
		self.list_labels = y_train

	def predict_vect_form(self,tweets):
		predictions = []
		for tweet in tweets:
			sims = self.index[self.tfidf[tweet]]
			sims = sorted(map(lambda x: (x[1],x[0]),enumerate(sims)),reverse=True)
			sims = sims[:self.k_neighbours]

			if self.counter_class == 1:
				print("Tweet: ")
				print(map(lambda tupla: (self.dict_id2word[tupla[0]], tupla[1]),tweet))
				for sim in sims:
					print("Train vect: ")
					print(map(lambda tupla: (self.dict_id2word[tupla[0]], tupla[1]),self.vectors_train[sim[1]]))
					print("Tag: "+ str(self.list_labels[sim[1]]))
			self.counter_class+=1

			if sims[0][0] == 0:
				predictions.append(0) 
			else:
				clases = map(lambda tupla: self.list_labels[tupla[1]],sims)
				classnfrec = np.unique(clases,return_counts=True)
				pos = np.argmax(classnfrec[1])
				predictions.append(classnfrec[0][pos])
		return np.array(predictions)