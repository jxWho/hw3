import numpy as np
import json
import gensim
from gensim.models import Word2Vec

data="data/hist_split.json"
fname="myModel"

with open( data ) as f:
    strings = f.read()
    jsonData = json.loads( strings )
    trainData = jsonData['train']
    sentences = []
    for td in trainData:
        ttd = td[0]
        temp = []
        for x in ttd:
            xx = x[0]
            if xx == None:
                xx = ""
            temp.append( xx )
        sentences.append(temp)

    model = Word2Vec(size=100, window=5, min_count=1)
    model.build_vocab(sentences)
    alpha, min_alpha, passes = (0.025, 0.001, 20)
    alpha_delta = (alpha - min_alpha) / passes

    for epoch in range( passes ):
        model.alpha, model.min_alpha = alpha, alpha
        model.train(sentences)

        print('completed pass %i at alpha %f' % (epoch + 1, alpha))
        alpha -= alpha_delta

        np.random.shuffle(sentences)

    model.save(fname)
