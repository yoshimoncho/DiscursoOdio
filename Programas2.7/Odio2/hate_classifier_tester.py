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
from sklearn.feature_selection import SelectKBest,SelectFdr,SelectFpr
from sklearn.feature_selection import chi2,mutual_info_classif,f_classif
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import matplotlib.pyplot as plt
import itertools
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import Normalizer,StandardScaler,RobustScaler
from neural_hate_classifier import neural_classifier


def test(select_features,norm = True,num_selected=1200,use_PCA = False,use_kbest = True):
    archivos = [r"training_set/hate_training_set.data"]
    tabla_err = []
    tabla_desv = []
    tablafinal = [tabla_err,tabla_desv]
    class_names = ["NoOdio","Odio"]
    min_media = 1
    oculta2 = 0
    oculta1 = 0
    #Solo una capa: 83 primera capa


    for archivo in archivos:

        datos = Datos(archivo,True)
        
        #270  o 1200 con bayes
        if select_features:
            if not use_PCA:
                if use_kbest:
                    selector = SelectKBest(chi2,num_selected)
                else:
                    #alpha=0.49
                    #alpha=0.46--
                    #alpha=0.35
                    #alpha=0.14
                    selector = SelectFpr(chi2,alpha=0.35)
                
                dd = seleccion_atributos(datos.datos[:,:-1],datos.datos[:,-1],datos.nombreAtributos[:-1],selector,verbose=True)
            else:
                pca = PCA(n_components=0.9,svd_solver="full")
                dd = pca.fit_transform(datos.datos[:,:-1])
                
        else:
            dd = datos.datos[:,:-1]

        
        if use_PCA or norm:
            X = dd
        else:
            encAtributos = preprocessing.OneHotEncoder(sparse=False)
            X = encAtributos.fit_transform(dd)
        Y=datos.datos[:,-1]

        #cv=KFold(len(X), n_folds=5, shuffle=True)
        print("Data shape: " + str(dd.shape))
        #cv = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
        cv = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
        err = []
        #Mejor encontrado 2.1e-06
        
        errores =  []
        nodos_intermedia = []
        for tasaaprend in range(1000,1010,10):
            for train_index, test_index in cv.split(X, Y):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                #MLPClassifier(hidden_layer_sizes=(270,135,66,33,7), max_iter = 20000,learning_rate_init=0.0001,verbose=True,tol=1e-3)
                #clf = KNeighborsClassifier(n_neighbors=5)
                #MultinomialNB(fit_prior=True,alpha=0.00000005)
                #MultinomialNB(fit_prior=True,alpha=0.00006)
                #DecisionTreeClassifier()
                #clf = MLPClassifier(hidden_layer_sizes=(203,153,115,87,65,49,),activation="relu")

                #clf = MLPClassifier(hidden_layer_sizes=(2000,),activation="logistic",max_iter=1000,learning_rate_init=0.001)
                clf = neural_classifier(X_train.shape[1],X_train.shape[0])
                #clf3 = BernoulliNB(fit_prior=True,alpha=0.00006)
                #clf = GaussianNB()

                #clf = Perceptron(n_iter=1000,eta0=0.01)
                #clf = MultinomialNB(fit_prior=True,alpha=2.1e-06)
                
                #pesos = map(lambda x: x+0 if x==1 else 1 ,Y_train)
                #clf = VotingClassifier(estimators=[('NB', clf1), ('MLP', clf2), ('BN', clf3)], voting='hard')
                #,sample_weight=map(lambda x: 1000 if x==0 else 1,Y_train)
                if norm:
                    #normalizador = Normalizer(norm="max")
                    #normalizador = StandardScaler()
                    normalizador = RobustScaler()
                    normalizador.fit(X_train)
                    X_train = normalizador.transform(X_train)
                    X_test = normalizador.transform(X_test)
                clf.fit(X_train, Y_train)
                err.append(1-clf.score(X_test, Y_test))
            
                        
            y_pred = clf.predict(X_test)
            media = np.mean(err)
            errores.append(media)
            nodos_intermedia.append(tasaaprend)
            print(media,np.std(err),tasaaprend)
        #plotgraph(nodos_intermedia,errores,"Constante Aprendizaje","Error")


        cnf_matrix = confusion_matrix(Y_test, y_pred)
        np.set_printoptions(precision=2)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')
        print("Last partition error: "+ str(1-clf.score(X_test, Y_test)))
        plt.show()
        plt.clf()

               




def seleccion_atributos(datos_X,datos_Y,features_names,selector,verbose=False):
    print("Initial shape: " + str(datos_X.shape))
    """
    sel = VarianceThreshold()
    datos_X = sel.fit_transform(datos_X)
    print("After VarianceThreshold: "+str(datos_X.shape))
    return datos_X
    """
    #print("First features filter: " + str(datos_X.shape))
    #datos_X = SelectKBest(chi2, k=550).fit_transform(datos_X,datos_Y )
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



def plotgraph(x,y,x_title,y_title):
    plt.plot(x, y)
    plt.ylabel(y_title)
    plt.xlabel(x_title)
    plt.show()
    plt.clf()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.

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





if __name__ == "__main__":
    test(select_features=False,norm=True,use_PCA = False,use_kbest=False)
