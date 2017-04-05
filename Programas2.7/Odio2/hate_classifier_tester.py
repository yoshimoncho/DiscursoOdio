import numpy as np
from supervisado.Datos import Datos
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
#from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,mutual_info_classif,f_classif
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import itertools


def test(select_features):
    archivos = [r"training_set/hate_training_set.data"]
    tabla_err = []
    tabla_desv = []
    tablafinal = [tabla_err,tabla_desv]
    class_names = ["NoOdio","Odio"]
    for archivo in archivos:
        print(archivo)
        datos = Datos(archivo,True)

        #270 con bayes
        if select_features:
            dd = seleccion_atributos(datos.datos[:,:-1],datos.datos[:,-1],datos.nombreAtributos[:-1],chi2,1200,verbose=True)
        else:
            dd = datos.datos[:,:-1]

        

        encAtributos = preprocessing.OneHotEncoder(sparse=False)
        X = encAtributos.fit_transform(dd)
        Y=datos.datos[:,-1]

        #cv=KFold(len(X), n_folds=5, shuffle=True)
        cv = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
        err = []
        for train_index, test_index in cv.split(X, Y):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            #datos.calcularMediasDesv(X_train)
            #X_train = datos.normalizarDatos(X_train)
            #X_test = datos.normalizarDatos(X_test)
            #MLPClassifier(hidden_layer_sizes=(270,135,66,33,7), max_iter = 20000,learning_rate_init=0.0001,verbose=True,tol=1e-3)
            #KNeighborsClassifier(n_neighbors=5)
            #MultinomialNB(fit_prior=True,alpha=0.00000005)
            #MultinomialNB(fit_prior=True,alpha=0.00006)
            # DecisionTreeClassifier()
            #MLPClassifier(hidden_layer_sizes=(1000,1000,1000),activation="relu",max_iter=10000, early_stopping=True)
            #clf = BernoulliNB(fit_prior=False,alpha=0.00006)
            #clf = MultinomialNB(fit_prior=True,alpha=0.00006)
            pesos = map(lambda x: x+0 if x==1 else 1 ,Y_train)
            clf.fit(X_train, Y_train,sample_weight=pesos)
            err.append(1-clf.score(X_test, Y_test))
        
        y_pred = clf.predict(X_test)
        cnf_matrix = confusion_matrix(Y_test, y_pred)
        np.set_printoptions(precision=2)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
        print("Last partition error: "+ str(1-clf.score(X_test, Y_test)))
        print(np.mean(err),np.std(err))
        plt.show()


def seleccion_atributos(datos_X,datos_Y,features_names,select_function,n_features=10,verbose=False):
    print("Initial shape: " + str(datos_X.shape))
    """
    sel = VarianceThreshold()
    datos_X = sel.fit_transform(datos_X)
    print("After VarianceThreshold: "+str(datos_X.shape))
    return datos_X
    """
    #print("First features filter: " + str(datos_X.shape))
    #datos_X = SelectKBest(chi2, k=550).fit_transform(datos_X,datos_Y )
    selector = SelectKBest(select_function,n_features)
    selector = selector.fit(datos_X,datos_Y)
    if verbose:
        f = open("training_set/filtered_features.txt","w")
        chosen_features = selector.get_support()
        chosen_names = []
        i = 0
        for i in range(len(features_names)):
            if chosen_features[i]:
                f.write(features_names[i]+",")
                chosen_names.append(features_names[i])
            i+=1
        f.write("STOP\n")
        f.close()
        print(chosen_names)
    datos_X = selector.transform(datos_X)
    print("Final shape: " + str(datos_X.shape))
    return datos_X

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




def full_set():
    archivo = r"ConjuntosDatos/xor.data"
    datos = Datos(archivo,True)
    encAtributos = preprocessing.OneHotEncoder(categorical_features=datos.nominalAtributos[:-1],sparse=False)
    X = encAtributos.fit_transform(datos.datos[:,:-1])
    Y=datos.datos[:,-1]
    clf = MultinomialNB() 
    #MLPClassifier(hidden_layer_sizes=(8,),activation='relu',solver='sgd',learning_rate_init=0.02,shuffle=False , max_iter=4000)
    clf.fit(X, Y)
    print(1-clf.score(X, Y))



if __name__ == "__main__":
    test(False)
