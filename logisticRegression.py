import theano
import theano.tensor as T

import numpy as np

class LogisticRegression:

    def __init__(self, dimension, num_classes):
        X = T.matrix('X')
        y = T.ivector('y')
        learning_rate = T.scalar('learning_rate')

        # initialize with 0 the weights W as a matrix
        self.W = theano.shared(value=np.zeros((dimension, num_classes),
                                                 dtype=theano.config.floatX),
                               name='W',
                               borrow=True)

        # initialize the biases b as a vector of 0s
        self.b = theano.shared(value=np.zeros((num_classes,),
                                                 dtype=theano.config.floatX),
                                name='b',
                                borrow=True)

        self.params = [self.W, self.b]

        #raise NotImplementedError

        #compile a theano function self.train that takes the matrix X, vector y, and learning rate as input

        #compile a theano function self.predict that takes a matrix X as input and returns a vector y of predictions


        self.p_y_given_x = T.nnet.softmax(T.dot(X, self.W) + self.b)

        self.y_pred = T.argmax( self.p_y_given_x, axis=1 )

        # keep track of model input
        self.input = X

        # cost function
        cost = self.negative_log_likelihood(y)
        # derivation
        g_W = T.grad(cost=cost, wrt=self.W)
        g_b = T.grad(cost=cost, wrt=self.b)
        # how to update
        updates = [(self.W, self.W - learning_rate * g_W),
                   (self.b, self.b - learning_rate * g_b)]
        # train model
        index = T.lscalar()

        self.train = theano.function(
                inputs=[X, y, learning_rate],
                updates=updates,
                outputs = cost
        )

        # predcit model
        self.predict = theano.function(
                inputs=[self.input],
                outputs=self.y_pred
        )

    # hack
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def fit(self, X, y, num_epochs=5, learning_rate=0.05, verbose=False):
        for i in range(num_epochs):
            cost = self.train(X, y, learning_rate)
            if verbose:
                print('cost at epoch {}: {}'.format(i, cost))

