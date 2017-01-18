import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from utils import tile_raster_images
import math
import matplotlib.pyplot as plt
import pickle

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
    mnist.test.labels

def weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias(shape):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial)

def sample_prob(probs):
    """Uzorkovanje vektora x prema vektoru vjerojatnosti p(x=1) = probs"""
    return tf.nn.relu(
        tf.sign(probs - tf.random_uniform(tf.shape(probs))))


def draw_weights(W, shape, N, interpolation="bilinear"):
    """Vizualizacija težina

    W -- vektori težina
    shape -- tuple dimenzije za 2D prikaz težina - obično dimenzije ulazne slike, npr. (28,28)
    N -- broj vektora težina
    """
    image = Image.fromarray( tile_raster_images(
        X=W.T,
        img_shape=shape,
        tile_shape=(int(math.ceil(N/20)), 20),
        tile_spacing=(1, 1)))
    plt.figure(figsize=(10, 14))
    plt.imshow(image, interpolation=interpolation)

def draw_reconstructions(ins, outs, states, shape_in, shape_state):
    """Vizualizacija ulaza i pripadajućih rekonstrkcija i stanja skrivenog sloja
    ins -- ualzni vektori
    outs -- rekonstruirani vektori
    states -- vektori stanja skrivenog sloja
    shape_in -- dimezije ulaznih slika npr. (28,28)
    shape_state -- dimezije za 2D prikaz stanja (npr. za 100 stanja (10,10)
    """
    plt.figure(figsize=(8, 12*4))
    for i in range(20):
        plt.subplot(20, 4, 4*i + 1)
        plt.imshow(ins[i].reshape(shape_in), vmin=0, vmax=1, interpolation="nearest")
        plt.title("Test input")
        plt.subplot(20, 4, 4*i + 2)
        plt.imshow(outs[i][0:784].reshape(shape_in), vmin=0, vmax=1, interpolation="nearest")
        plt.title("Reconstruction")
        plt.subplot(20, 4, 4*i + 3)
        plt.imshow(states[i][0:(shape_state[0] * shape_state[1])].reshape(shape_state), vmin=0, vmax=1, interpolation="nearest")
        plt.title("States")
    plt.tight_layout()


Nv = 784
v_shape = (28,28)
Nh = 100
h1_shape = (10,10)
Nh2 = 100
h2_shape = (10,10)

gibbs_sampling_steps = 4
alpha = 0.1

g2 = tf.Graph()
with g2.as_default():
    X2 = tf.placeholder("float", [None, Nv])
    w1s, vb1s, hb1s = pickle.load(open("weights.pickle", "rb"))
    w1a = tf.Variable(w1s)
    vb1a = tf.Variable(vb1s)
    hb1a = tf.Variable(hb1s)
    w2 = weights([Nh, Nh2])
    vb2 = bias([Nh])
    hb2 = bias([Nh2])

    # vidljivi sloj drugog RBM-a
    v2_prob = tf.nn.softmax(tf.matmul(X2, w1a) + hb1a)
    v2 = sample_prob(v2_prob)
    # skriveni sloj drugog RBM-a
    h2_prob = tf.nn.softmax(tf.matmul(v2, w2) + hb2)
    h2 = sample_prob(h2_prob)
    h3 = h2

    for step in range(gibbs_sampling_steps):
        v3_prob = tf.nn.softmax(tf.matmul(h3, w2, transpose_b=True) + vb2)
        v3 = sample_prob(v3_prob)
        h3_prob = tf.nn.softmax(tf.matmul(v3, w2) + hb2)
        h3 = sample_prob(h3_prob)

    # print('vb1a', vb1a.get_shape())
    # print('hb1a', hb1a.get_shape())
    # print('w1', w1a.get_shape())
    # print('w2', w2.get_shape())
    # print('v2', v2.get_shape())
    # print('h2', h2.get_shape())
    # print('v3', v3.get_shape())
    # print('h3', h3.get_shape())

    w2_positive_grad = tf.matmul(v2, h2, transpose_a=True)
    w2_negative_grad = tf.matmul(v3, h3, transpose_a=True)

    dw2 = (w2_positive_grad - w2_negative_grad) / tf.to_float(tf.shape(v2)[0])

    update_w2 = tf.assign_add(w2, alpha * dw2)
    update_vb2 = tf.assign_add(vb2, alpha * tf.reduce_mean(v2 - v3, 0))
    update_hb2 = tf.assign_add(hb2, alpha * tf.reduce_mean(h2 - h3, 0))

    out2 = (update_w2, update_vb2, update_hb2)

    # rekonsturkcija ulaza na temelju krovnog skrivenog stanja h3
    v4_prob = tf.nn.softmax(tf.matmul(h3, w2, transpose_b=True) + hb1a)
    v4 = sample_prob(v4_prob)
    v5_prob = tf.nn.softmax(tf.matmul(v4, w1a, transpose_b=True) + vb1a)

    err2 = X2 - v5_prob
    err_sum2 = tf.reduce_mean(err2 * err2)

    initialize2 = tf.initialize_all_variables()

batch_size = 100
epochs = 100
n_samples = mnist.train.num_examples

total_batch = int(n_samples / batch_size) * epochs

with tf.Session(graph=g2) as sess:
    sess.run(initialize2)

    for i in range(total_batch):
        batch, label = mnist.train.next_batch(batch_size)
        err, _ = sess.run([err_sum2, out2], feed_dict={X2: batch})

        if i%(int(total_batch/10)) == 0:
            print("Batch count: ", i, "  Avg. reconstruction error: ", err)

    w2s, vb2s, hb2s = sess.run([w2, vb2, hb2], feed_dict={X2: batch})
    vr2, h3_probs, h3s = sess.run([v5_prob, h3_prob, h3], feed_dict={X2: teX[0:50,:]})

# vizualizacija težina
draw_weights(w2s, h1_shape, Nh2, interpolation="nearest")

# vizualizacija rekonstrukcije i stanja
draw_reconstructions(teX, vr2, h3s, v_shape, h2_shape)
plt.show()
