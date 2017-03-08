from gensim import corpora




def main():
	dictionary = corpora.Dictionary()
	dictionary = dictionary.load('tweets.dict')
	new_vec = dictionary.doc2bow("caranchoa".split())
	print(new_vec)


def guardado():

	texts = []
	for j in range(1,4):

		file = open("treatcorpus/treat_basegrande"+str(j)+"_clean.txt","r")
		contador = 0 
		for line in file:
			
			if line != "\n":
				if line[-1] != "\n":
					print("Ojo al corte")
				texts.append(line[0:-1].split())
			else:
				contador +=1



	dictionary = corpora.Dictionary(texts)
	dictionary.save('tweets.dict')





if __name__ == "__main__":
	main()