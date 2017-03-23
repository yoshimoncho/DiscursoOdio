import numpy as np
import gensim


def main2():
	numpy_matrix = np.random.randint(10, size=[5,2])  # random matrix as an example
	print(numpy_matrix)
	corpus = gensim.matutils.Dense2Corpus(numpy_matrix)
	numpy_matrix = gensim.matutils.corpus2dense(corpus, num_terms=5)
	print(numpy_matrix)






def main():
	file = open("treatcorpus/treat_basegrande1_clean.txt","r")
	for line in file:

		if line != "\n":
			if line[-1] != "\n":
				print("Ojo al corte")
			cadena = line[:-1]
			print(cadena)
			texts.append(cadena)
		else:
			contador +=1
	file.close()



def guardado():
	fout = open("heap.txt","w")
	total_term = 0

	texts = []
	contador = 0 
	for j in range(1,11):

		file = open("treatcorpus/treat_basegrande"+str(j)+"_clean.txt","r")
		
		for line in file:
			
			if line != "\n":
				if line[-1] != "\n":
					print("Ojo al corte")

				aux = line[0:-1].split()
				total_term += len(aux)
				texts.append(aux)
				dictionary = corpora.Dictionary(texts)
				contador +=1
				if contador >= 1000:
					contador = 0
					fout.write(str(total_term) +" "+str(len(dictionary.token2id))+"\n")




	dictionary = corpora.Dictionary(texts)
	dictionary.save('tweets.dict')

if __name__ == "__main__":
	main2()