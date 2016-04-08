import matplotlib.pyplot as plt
import numpy as np
import sklearn
from dependencyRNN import DependencyRNN
from sklearn.manifold import TSNE

fileName = "random_init.npz"
answerEmbedding = DependencyRNN.load(fileName).answers

temp = []
for x in answerEmbedding:
    temp.append( answerEmbedding[x] )

X = np.asarray(temp)
tsne = TSNE(n_components=2, perplexity=30.0)
X_reduced = tsne.fit_transform(X)

print X_reduced

plt.plot(X_reduced)
