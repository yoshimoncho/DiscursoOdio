from gensim.models.doc2vec import Doc2Vec as d2v
from LabeledLineSentence import LabeledLineSentence
import numpy as np
import time


class Doc2Vec(object):
    """docstring for Doc2Vec"""

    def __init__(self):
        super(Doc2Vec, self).__init__()
        self.doc2vec = None

    def train(self, input_path, save_location, dimension=50, epochs=20, method="DBOW", ides="Number"):
        sentences = LabeledLineSentence(input_path, ides)

        total_start = time.time()
        dm_ = 1
        if method == "DBOW":
            dm_ = 0
        model = d2v(min_count=1, window=7, size=dimension, dm=dm_, sample=1e-3, negative=5, workers=6,
                                      alpha=0.02, )

        print "inicio vocab"
        model.build_vocab(sentences)
        sentences.reloadDoc()
        print "fin vocab"

        model.train(sentences, total_examples=model.corpus_count, epochs=epochs)

        model.save(save_location)

        total_end = time.time()

        print "tiempo total:" + str((total_end - total_start) / 60.0)

    def simulateVectorsFromVectorText(self, vectorText, modelLocation=None):
        if self.doc2vec is None and modelLocation is None:
            raise Exception("Se tiene que cargar el modelo")

        if self.doc2vec is None:
            self.doc2vec = d2v.load(modelLocation)

        vector = np.array(self.doc2vec.infer_vector(vectorText, steps=3, alpha=0.1))
        return vector / np.linalg.norm(vector)

    def loadModel(self, modelLocation):
        self.doc2vec = d2v.load(modelLocation)

    def getNormalizedTagsVectors(self):
        doctags = self.doc2vec.docvecs.doctags
        ret = {}
        for doctag in doctags:
            vector = np.array(self.doc2vec.docvecs[doctag])
            ret[doctag] = vector / np.linalg.norm(vector)

        return ret