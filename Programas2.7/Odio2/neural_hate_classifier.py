from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Dropout
from keras.layers.recurrent import SimpleRNN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import datasets, metrics
from keras.utils import plot_model



"""
#(0.14078574180603026, 0.014518592344265819, 1000)
#TF_IDF, usando los mismos atributos seleccionados por 1 y 0
# Normalizacion robusta, (recorrido intercuartilico y mediana)

def neural_classifier(n_input,n_batch):

	def create_model():
		model = Sequential()
		model.add(Dense(1600, input_dim=n_input, activation='relu'))
		model.add(Dropout(rate=0.8))
		model.add(Dense(1600, activation='relu'))
		model.add(Dropout(rate=0.8))
		model.add(Dense(1600, activation='relu'))
		model.add(Dropout(rate=0.85))
		#model.add(Dense(1400, activation='relu'))
		#model.add(Dropout(rate=0.85))
		model.add(Dense(1, activation='sigmoid'))
		# Compile model
		model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
		return model

	return KerasClassifier(build_fn=create_model, epochs=100, batch_size=n_batch, verbose=0)
"""




"""
#Normalizacion l2
def neural_classifier(n_input,n_batch):

	def create_model():
		model = Sequential()
		model.add(Dense(350, input_dim=n_input, activation='relu'))
		model.add(Dropout(rate=0.75))
		model.add(Dense(350, activation='relu'))
		model.add(Dropout(rate=0.75))
		model.add(Dense(350, activation='relu'))
		model.add(Dropout(rate=0.75))
		model.add(Dense(350, activation='relu'))
		model.add(Dropout(rate=0.75))
		model.add(Dense(1, activation='sigmoid'))
		# Compile model
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model

	return KerasClassifier(build_fn=create_model, epochs=500, batch_size=n_batch, verbose=1)
"""



def neural_classifier(n_input,n_batch):

	def create_model():
		model = Sequential()
		model.add(Dense(1600, input_dim=n_input, activation='relu'))
		model.add(Dropout(rate=0.8))
		model.add(Dense(1600, activation='relu'))
		model.add(Dropout(rate=0.8))
		model.add(Dense(1600, activation='relu'))
		model.add(Dropout(rate=0.85))
		#model.add(Dense(1400, activation='relu'))
		#model.add(Dropout(rate=0.85))
		model.add(Dense(1, activation='sigmoid'))
		# Compile model
		model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
		return model

	return KerasClassifier(build_fn=create_model, epochs=100, batch_size=n_batch, verbose=1)



def plot_neural_model():
	model = Sequential()
	model.add(Dense(1600, input_dim=2532, activation='relu'))
	model.add(Dropout(rate=0.8))
	model.add(Dense(1600, activation='relu'))
	model.add(Dropout(rate=0.8))
	model.add(Dense(1600, activation='relu'))
	model.add(Dropout(rate=0.85))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
	plot_model(model, to_file='hate_neural_network.png')






if __name__ == "__main__":
	main()
	#plot_neural_model()
