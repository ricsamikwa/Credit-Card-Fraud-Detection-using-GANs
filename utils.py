
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# initialize the weight-matrix W.
def weight_matrix_init(size): 

    in_dim = size[0]
    weight_matrix_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=weight_matrix_stddev)

# input, uniform sampling
def sampleY(m, n):   
    return np.random.uniform(-1., 1., size=[m, n])

#one hot encoder for processing categorical data
def one_hot_encoder(x, depth):

    encoded = np.zeros((len(x), depth), dtype=np.int32)
    x = x.astype(int)

    for i in range(encoded.shape[0]):
        encoded[i, x[i]] = 1
    return encoded


def sample_shuffle(X, labels):
	n_samples = len(X)
	s = np.arange(n_samples)
	np.random.shuffle(s)
	return np.array(X[s]), labels[s]

def sample_shuffle_uspv(X):
	n_samples = len(X)
	s = np.arange(n_samples)
	np.random.shuffle(s)
	return np.array(X[s])



def draw_trends(fm_loss, f1):

    fig = plt.figure()
    fig.patch.set_facecolor('w')
    p4, = plt.plot(fm_loss, "-b")
    plt.xlabel("# of epoch")
    plt.ylabel("feature matching loss")

    fig = plt.figure()
    fig.patch.set_facecolor('w')
    p5, = plt.plot(f1, "-y")
    plt.xlabel("# of epoch")
    plt.ylabel("F1")
    plt.show()
 

def pull_away_term(g):

    Nor = tf.norm(g, axis=1)
    Nor_mat = tf.tile(tf.expand_dims(Nor, axis=1),
                      [1, tf.shape(g)[1]])
    X = tf.divide(g, Nor_mat)
    X_Square= tf.square(tf.matmul(X, tf.transpose(X)))
    mask = tf.subtract(tf.ones_like(X_Square),
                       tf.diag(
                           tf.ones([tf.shape(X_Square)[0]]))
                       )
    pt_loss = tf.divide(tf.reduce_sum(tf.multiply(X_Square, mask)),
                        tf.multiply(
                            tf.cast(tf.shape(X_Square)[0], tf.float32),
                            tf.cast(tf.shape(X_Square)[0]-1, tf.float32)))

    return pt_loss


