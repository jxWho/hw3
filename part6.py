import matplotlib.pyplot as plt
import numpy as np
import sklearn
from dependencyRNN import DependencyRNN
from sklearn.manifold import TSNE

fileName = "random_init.npz"
answerEmbedding = DependencyRNN.load(fileName).answers

temp = []
n = []
for x in answerEmbedding:
    temp.append( answerEmbedding[x] )
    n.append( x )

X = np.asarray(temp)
tsne = TSNE(n_components=2, perplexity=30.0)
X_reduced = tsne.fit_transform(X)

xs= []
ys = []
for x, y in X_reduced:
    xs.append(x)
    ys.append(y)

fig, ax = plt.subplots()
ax.scatter(xs, ys)
for i, txt in enumerate(n):
    ax.annotate(txt, (xs[i], ys[i]))

plt.show()
'''
plt.plot(X_reduced)
plt.ylabel('some numbers')
plt.show()
'''
