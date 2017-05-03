
from hate_classifier import Hate_classifier
import codecs
import numpy as np



# Permite ver la efectividad del clasificador , en este caso se comprueba
# que los tweets clasificados como odio, verdaderamente lo sean (es un archivo muy grande)

def main():
    clf = Hate_classifier(pre_filter=True,using_bigrams=False)
    clf.fit(norm=True,clf_name = "neural")

    file = codecs.open("validation_set/basegrande15.txt", "r", "utf-8")
    fout = codecs.open("validation_set/basegrande15_3000_neural_noprefilter.txt", "w", "utf-8")
    lista_nofiltrada = []

    for line in file:
        frags = line.split(";||;")
        if len(frags) != 6:
            print("Error en el numero de campos")
        else:
            texto = frags[4]
            prediccion = clf.predict(texto)[0]
            if prediccion == 1:
                fout.write(texto+";||;1;||;\n")
            elif prediccion == 2:
                pass
                #fout.write(texto+";||;0;||;\n")
            else:
                pass
                #lista_nofiltrada.append(texto)

    """

    indexes = sorted(np.random.permutation(len(lista_nofiltrada))[:6000])
    for index in indexes:
        fout.write(lista_nofiltrada[index]+";||;\n")
    """

    file.close()
    fout.close()




def put_real_labels():
    file = codecs.open("validation_set/basegrande15_3000_2.txt", "r", "utf-8")
    file2 = codecs.open("validation_set/basegrande15_3000_bayes.txt", "r", "utf-8")
    fout = codecs.open("validation_set/basegrande15_3000_bayes_2.txt", "w", "utf-8")
    for line in file:
        frags = line.split(";||;")
        line_out = file2.readline()[:-1]
        fout.write(line_out+frags[2])

    file.close()
    file2.close()
    fout.close()







def aux_main():
    clf = Hate_classifier(pre_filter=True,using_bigrams=False)
    clf.fit(norm=False,clf_name = "bayes")
    file = codecs.open("validation_set/nofiltrados_basegrande15_tagged.txt", "r", "utf-8")
    fout = codecs.open("validation_set/nofiltrados_basegrande15_neural.txt", "w", "utf-8")
    lista_nofiltrada = []

    for line in file:
        frags = line.split(";||;")
        if len(frags) != 2:
            print("Error en el numero de campos")
        else:
            texto = frags[0]
            prediccion = clf.predict(texto)[0]
            if prediccion == 1:
                fout.write(texto+";||;1;||;\n")
            elif prediccion == 2:
                fout.write(texto+";||;0;||;\n")
            else:
                pass
                #lista_nofiltrada.append(texto)

    """

    indexes = sorted(np.random.permutation(len(lista_nofiltrada))[:6000])
    for index in indexes:
        fout.write(lista_nofiltrada[index]+";||;\n")
    """

    file.close()
    fout.close()


if __name__ == "__main__":
    main()
    #put_real_labels()
    #aux_main()