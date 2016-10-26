import tensorflow as tf

class TFLogReg:
    def __init__(self, D, C, param_delta=0.5, param_lambda=0.1):
        self.X = tf.placeholder(tf.float32, [None, D])
        self.Yoh_ = tf.placeholder(tf.float32, [None, C])
        self.W = tf.Variable(tf.zeros([D, C]))
        self.b = tf.Variable(tf.zeros([C]))

        self.probs = tf.nn.softmax(tf.matmul(self.X, self.W) + self.b)
        self.loss = tf.reduce_mean(-tf.reduce_sum(tf.mul(tf.log(self.probs), self.Yoh_), reduction_indices=1)) + param_lambda * tf.nn.l2_loss(self.W)
        trainer = tf.train.GradientDescentOptimizer(param_delta)
        self.train_step = trainer.minimize(self.loss)
        self.session = tf.Session()

    def train(self, X, Yoh_, param_niter):
        self.session.run(tf.initialize_all_variables())
        for i in range(param_niter):
            loss, _, W, b = self.session.run([self.loss, self.train_step, self.W, self.b],
                                            feed_dict={self.X: X, self.Yoh_: Yoh_})
            if i % 10 == 0:
                print(i, loss)

    def eval(self, X):
        return self.session.run([self.probs], feed_dict={self.X: X})
