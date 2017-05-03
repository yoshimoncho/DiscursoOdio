from gensim import corpora, models, similarities
from itertools import imap
from collections import defaultdict
from training_set_generator import load_filtered_features
import pickle




def generate_vocab(corpus_tweets_file,tagged_tweets_file):
	corpus_with_ids = corpora.MmCorpus(corpus_tweets_file)
	corpus = map(lambda tweet: tweet[1:],corpus_with_ids)
	vocab_features = pickle.load( open( "training_set/vocab_features.p", "rb" ) )
	dic = pickle.load( open( "program_data/dic_id2word.p", "rb" ) )
	filtered_features = load_filtered_features()

	new_hate_wordsfreq = [defaultdict(lambda:0),defaultdict(lambda:0)]


	list_tuples_labels = extract_list_tuple_labels(tagged_tweets_file)

	pos_relativa = 0
	hate_indexes = []

	while len(list_tuples_labels) > 0:
		if(list_tuples_labels[0][0]== abs(corpus_with_ids[pos_relativa][0][0])):
			tupla = list_tuples_labels.pop(0)
			for tupla_tweet in corpus_with_ids[pos_relativa][1:]:
				# Dependiendo de si el tweet es de odio o no se aumenta la freq en 1 del termino
				new_hate_wordsfreq[tupla[1]][tupla_tweet[0]]+=1
		pos_relativa+=1

	n_new_words = 0

	for term in filtered_features:
		freq_hate = new_hate_wordsfreq[1][term]
		freq_neutral = new_hate_wordsfreq[0][term]
		if (freq_hate > freq_neutral) and  (term not in vocab_features):
			print("freq_hate: "+str(freq_hate), "freq_neutral: "+str(freq_neutral),dic[term])
			n_new_words += 1

	print("Number of new words: "+str(n_new_words))





def extract_list_tuple_labels(filename):
	f = open(filename,"r")
	#Diccionario de id:tweet => etiqueta
	labels_list = []
	n_id = 0
	for line in f:
		frags = line.split(";||;")
		if len(frags) != 3:
			print("Wrong format, maybe some labels are lost!")
			exit(-1)
		labels_list.append((int(frags[0][3:]),int(frags[2])))

	print("Number of labels: "+ str(len(labels_list)))
	f.close()
	return labels_list


if __name__ == "__main__":
	generate_vocab("program_data/tweets_corpus.mm","training_set/tagged_1000.txt")