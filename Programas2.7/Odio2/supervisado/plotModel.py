import matplotlib.pyplot as plt
import numpy as np
import itertools
# Modificada Juan Carlos y Ana
def plotModel(ev_best,ev_media):
    #Genera una gráfica
    plt.figure(figsize=((15,5)))
    plt.hold(True)
    plt.plot(ev_best, 'bo-',label ='Mejor')
    plt.plot(ev_media, 'ro-',label ='Media')
    plt.legend(loc=4)
    plt.xlabel('Generacion')
    plt.ylabel('Fitness')
    plt.ylim(np.min(ev_best+ev_media)*0.95,np.max(ev_best+ev_media)*1.05)
    plt.show()


def plot_confusion_matrix(conf_arr, labels):
    #Genera una matriz de confusión
    conf_arr=np.array(conf_arr)

    norm_conf = []
    
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                    interpolation='nearest')

    width, height = conf_arr.shape

    for x,y in itertools.product(range(width), range(height)):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    plt.xticks(range(width), labels )
    plt.yticks(range(height), labels )
    plt.show()
    
