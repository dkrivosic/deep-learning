import tensorflow as tf
import numpy as np

class TFDeep:
    def __init__(self, layers, param_delta=0.5, param_lambda=0.1):
        D = layers[0]
        C = layers[-1]
        self.X = tf.placeholder(tf.float32, [None, D])
        self.Yoh_ = tf.placeholder(tf.float32, [None, C])
        self.h = layers
        self.W = []
        self.b = []
        for i in range(1, len(self.h)):
            self.W.append(tf.Variable(tf.random_normal([self.h[i-1], self.h[i]]), name='W'+str(i)))
            self.b.append(tf.Variable(tf.random_normal([self.h[i]]), name='b'+str(i)))

        a = self.X
        for i in range(len(self.W) - 1):
            a = tf.nn.relu(tf.matmul(a, self.W[i]) + self.b[i])
        self.probs = tf.nn.softmax(tf.matmul(a, self.W[-1]) + self.b[-1])

        regularization_loss = 0
        for w in self.W:
            regularization_loss += tf.nn.l2_loss(w)
        self.loss = tf.reduce_mean(-tf.reduce_sum(tf.mul(tf.log(self.probs), self.Yoh_), reduction_indices=1)) + param_lambda * regularization_loss

        trainer = tf.train.GradientDescentOptimizer(param_delta)
        self.train_step = trainer.minimize(self.loss)
        self.session = tf.Session()

    def train(self, X, Yoh_, param_niter):
        self.session.run(tf.initialize_all_variables())
        for i in range(param_niter):
            loss = self.session.run([self.loss, self.train_step] + self.W + self.b,
                                            feed_dict={self.X: X, self.Yoh_: Yoh_})[0]
            if i % 10 == 0:
                print(i, loss)

    def eval(self, X):
        return self.session.run([self.probs], feed_dict={self.X: X})[0]

    def count_params(self):
        for v in tf.trainable_variables():
            print(v.name)
        total_count = 0
        for i in range(1, len(self.h)):
            total_count += self.h[i] * self.h[i-1]
        total_count += sum(self.h[1:])
        print("Total parameter count: " + str(total_count))
