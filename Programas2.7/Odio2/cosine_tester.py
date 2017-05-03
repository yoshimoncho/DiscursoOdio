import numpy as np
from sklearn.model_selection import StratifiedKFold
from gensim import corpora
from cosine_knn import Cosine_Hate_Classifier
from sklearn.metrics import accuracy_score







def test_cosine():
	
	list_tuples_labels = extract_list_tuple_labels("training_set/tagged_3000.txt")
	corpus = corpora.MmCorpus("program_data/tweets_corpus.mm")

	aux = []
	selected_corpus = []
	i = 0
	while len(list_tuples_labels) > 0:
		if(list_tuples_labels[0][0]== abs(corpus[i][0][0])):
			aux.append(list_tuples_labels.pop(0)[1])
			selected_corpus.append(corpus[i][1:])
		i += 1

	list_tuples_labels = np.array(aux)
	selected_corpus = np.array(selected_corpus)
	cl = Cosine_Hate_Classifier("program_data/tweets_corpus.mm")
	error = []
	cv = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
	counter = 0
	for train_index, test_index in cv.split(np.zeros(len(aux)), list_tuples_labels):
		if counter == 0:
			print(train_index[0])
		counter += 1
		cl.fit(selected_corpus[train_index],list_tuples_labels[train_index])
		resultado = cl.predict_vect_form(selected_corpus[test_index])
		error.append(accuracy_score(list_tuples_labels[test_index],resultado))
	print(np.mean(error))
		





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
	test_cosine()