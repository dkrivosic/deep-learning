import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

DATA_DIR = '/Users/domagoj/fer/deep-learning/lab_2/data'
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
num_classes = 10
weight_decay = 0.1
batch_size = 50
epochs = 100

y_ = tf.placeholder(tf.float32, [None, num_classes])
x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, 16])
b_conv1 = bias_variable([16])

h_1 = tf.nn.relu(max_pool_2x2(conv2d(x_image, W_conv1) + b_conv1))

W_conv2 = weight_variable([5, 5, 16, 32])
b_conv2 = bias_variable([32])

h_2 = tf.nn.relu(max_pool_2x2(conv2d(h_1, W_conv2) + b_conv2))

h_2_flat = tf.reshape(h_2, [-1, 7 * 7 * 32])

W_fc1 = weight_variable([ 7 * 7 * 32, 512])
b_fc1 = bias_variable([512])

h_3 = tf.nn.relu(tf.matmul(h_2_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([512, num_classes])
b_fc2 = bias_variable([num_classes])
y = tf.matmul(h_3, W_fc2) + b_fc2

# Training
with tf.Session() as sess:
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_) +
        weight_decay * tf.nn.l2_loss(W_conv1) + weight_decay * tf.nn.l2_loss(W_conv2) +
        weight_decay * tf.nn.l2_loss(W_fc1) + weight_decay * tf.nn.l2_loss(W_fc2))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())

    for epoch in range(epochs):
        print("Epoch %d"%(epoch))
        num_batches = mnist.train.num_examples // batch_size
        for i in range(num_batches):
            batch = mnist.train.next_batch(batch_size)
            if (i+1)%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
                print("step %d/%d, training accuracy %g"%(i+1, mnist.train.num_examples//batch_size, train_accuracy))
                train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels}))
