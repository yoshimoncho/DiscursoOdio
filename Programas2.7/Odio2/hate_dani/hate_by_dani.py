import codecs
from limpiador import Limpiador
from CleanText import CleanText

if __name__ == '__main__':
    tweets = []
    tweet = {"text": "", "class": 0}
    #lectura
    for line in codecs.open('training_set/tagged_1000.txt', 'r', 'utf-8'):
        id, text, c = line.split(";||;")
        tweet_copy = tweet.copy()
        tweet_copy["text"] = text
        tweet_copy["class"] = int(c)
        tweets.append(tweet_copy)
    #limpieza
    for tweet in tweets:
        tweet["text"] = Limpiador.clean(tweet["text"])
    #stopwords
    for tweet in tweets:
        tweet["text"] = CleanText.stopWordsByLanguagefilter(tweet["text"], 'es')

    #lematizacion
    for tweet in tweets:
        tweet["text"] = CleanText.stemmingByLanguage(tweet["text"], 'es')

    # ocurrences
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    text_data = [tweet["text"] for tweet in tweets]

    X_train_counts = count_vect.fit_transform(text_data)
    print X_train_counts.shape
    from sklearn.feature_extraction.text import TfidfTransformer

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    from sklearn.model_selection import train_test_split

    classes = [tweet["class"] for tweet in tweets]
    X_train_tfidf, X_test, classes, y_test = train_test_split(X_train_tfidf, classes, test_size=0.10, random_state=42)

    print "nb MultinomialNB"
    from sklearn.naive_bayes import MultinomialNB, GaussianNB
    clf = MultinomialNB().fit(X_train_tfidf, classes)

    predicted = clf.predict(X_test)
    predicted_array = []
    num_predicted = len(predicted)
    predicted_array.append(predicted)
    from sklearn import metrics
    print metrics.confusion_matrix(y_test, predicted)
    print metrics.accuracy_score(y_test, predicted)

    print "KNN"
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean').fit(X_train_tfidf, classes)

    predicted = clf.predict(X_test)
    predicted_array.append(predicted)
    print metrics.confusion_matrix(y_test, predicted)
    print metrics.accuracy_score(y_test, predicted)

    print "SDG"
    from sklearn.linear_model import SGDClassifier

    clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=100, random_state = 41).fit(X_train_tfidf, classes)
    predicted = clf.predict(X_test)
    predicted_array.append(predicted)
    print metrics.confusion_matrix(y_test, predicted)
    print metrics.accuracy_score(y_test, predicted)

    print "NN"
    from sklearn.neural_network.multilayer_perceptron import MLPClassifier
    predicted_array.append(predicted)
    clf = MLPClassifier(hidden_layer_sizes=(2,)).fit(X_train_tfidf, classes)
    predicted = clf.predict(X_test)

    print metrics.confusion_matrix(y_test, predicted)
    print metrics.accuracy_score(y_test, predicted)


    from Doc2vec import Doc2Vec
    with codecs.open('training_set/dani.txt', 'w', 'utf-8') as f_out:
        for i, tweet in enumerate(tweets):
            f_out.write(str(i) + "\n")
            f_out.write(tweet["text"] + "\n")

    d2v = Doc2Vec()
    d2v.train('training_set/dani.txt', 'training_set/model.d2v', dimension=200, epochs=20)
    d2v = Doc2Vec()
    d2v.loadModel('training_set/model.d2v')
    vectors = d2v.getNormalizedTagsVectors()
    vectors_array = []
    for i, tweet in enumerate(tweets):
        vectors_array.append(vectors[str(i)])
    classes = [tweet["class"] for tweet in tweets]
    train, X_test, classes, y_test = train_test_split(vectors_array, classes, test_size=0.10, random_state=41)

    print "MLP doc2vec"
    clf = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic').fit(train, classes)

    predicted = clf.predict(X_test)
    print metrics.confusion_matrix(y_test, predicted)
    print metrics.accuracy_score(y_test, predicted)

    print "knn doc2vec"
    clf = KNeighborsClassifier(n_neighbors=7).fit(train, classes)

    predicted = clf.predict(X_test)
    print metrics.confusion_matrix(y_test, predicted)
    print metrics.accuracy_score(y_test, predicted)

    print "GaussianNB doc2vec"
    clf = GaussianNB().fit(train, classes)

    predicted = clf.predict(X_test)
    print metrics.confusion_matrix(y_test, predicted)
    print metrics.accuracy_score(y_test, predicted)







