
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import pull_away_term, one_hot_encoder, weight_matrix_init, sample_shuffle, sample_shuffle_uspv, sampleY, draw_trends
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import os
import sys
 
dra_tra_pro = int(sys.argv[1])  #input determines whether we display training evaluation metrics or classification report

minibatch_size = 70
input_size = 50

Dis_dim = [input_size, 100, 50, 2]
Gen_dim = [50, 100, input_size]
noise_dim = Gen_dim[0]


# placeholders for labeled data, unlabeled data, noise data and target data.

X_oc = tf.placeholder(tf.float32, shape=[None, input_size])
Z = tf.placeholder(tf.float32, shape=[None, noise_dim])
X_tar = tf.placeholder(tf.float32, shape=[None, input_size])

# weights and biases of generator.

Gen_W1 = tf.Variable(weight_matrix_init([Gen_dim[0], Gen_dim[1]]))
Gen_b1 = tf.Variable(tf.zeros(shape=[Gen_dim[1]]))

Gen_W2 = tf.Variable(weight_matrix_init([Gen_dim[1], Gen_dim[2]]))
Gen_b2 = tf.Variable(tf.zeros(shape=[Gen_dim[2]]))

parameters_Gen = [Gen_W1, Gen_W2, Gen_b1, Gen_b2]

# weights and biases of discriminator.

Dis_W1 = tf.Variable(weight_matrix_init([Dis_dim[0], Dis_dim[1]]))
Dis_b1 = tf.Variable(tf.zeros(shape=[Dis_dim[1]]))

Dis_W2 = tf.Variable(weight_matrix_init([Dis_dim[1], Dis_dim[2]]))
Dis_b2 = tf.Variable(tf.zeros(shape=[Dis_dim[2]]))

Dis_W3 = tf.Variable(weight_matrix_init([Dis_dim[2], Dis_dim[3]]))
Dis_b3 = tf.Variable(tf.zeros(shape=[Dis_dim[3]]))

parameters_Dis = [Dis_W1, Dis_W2, Dis_W3, Dis_b1, Dis_b2, Dis_b3]

# weights and biases of pre-train net for density estimation.

T_W1 = tf.Variable(weight_matrix_init([Dis_dim[0], Dis_dim[1]]))
T_b1 = tf.Variable(tf.zeros(shape=[Dis_dim[1]]))

T_W2 = tf.Variable(weight_matrix_init([Dis_dim[1], Dis_dim[2]]))
T_b2 = tf.Variable(tf.zeros(shape=[Dis_dim[2]]))

T_W3 = tf.Variable(weight_matrix_init([Dis_dim[2], Dis_dim[3]]))
T_b3 = tf.Variable(tf.zeros(shape=[Dis_dim[3]]))

parameters_T = [T_W1, T_W2, T_W3, T_b1, T_b2, T_b3]


def generator(z):
    Gen_h1 = tf.nn.relu(tf.matmul(z, Gen_W1) + Gen_b1)
    Gen_Logit = tf.nn.tanh(tf.matmul(Gen_h1, Gen_W2) + Gen_b2)
    return Gen_Logit


def discriminator(x):
    Dis_h1 = tf.nn.relu(tf.matmul(x, Dis_W1) + Dis_b1)
    Dis_h2 = tf.nn.relu(tf.matmul(Dis_h1, Dis_W2) + Dis_b2)
    Dis_Logit = tf.matmul(Dis_h2, Dis_W3) + Dis_b3
    Dis_prob = tf.nn.softmax(Dis_Logit)
    return Dis_prob, Dis_Logit, Dis_h2


# pre-train net for density estimation.

def discriminator_t(x):
    T_h1 = tf.nn.relu(tf.matmul(x, T_W1) + T_b1)
    T_h2 = tf.nn.relu(tf.matmul(T_h1, T_W2) + T_b2)
    T_logit = tf.matmul(T_h2, T_W3) + T_b3
    T_prob = tf.nn.softmax(T_logit)
    return T_prob, T_logit, T_h2


Dis_prob_real, Dis_logit_real, Dis_h2_real = discriminator(X_oc)

G_sample = generator(Z)
Dis_prob_gen, Dis_logit_gen, Dis_h2_gen = discriminator(G_sample)

Dis_prob_tar, Dis_logit_tar, Dis_h2_tar = discriminator_t(X_tar)
Dis_prob_tar_gen, Dis_logit_tar_gen, Dis_h2_tar_gen = discriminator_t(G_sample)


# discriminator loss
y_real= tf.placeholder(tf.int32, shape=[None, Dis_dim[3]])
y_gen = tf.placeholder(tf.int32, shape=[None, Dis_dim[3]])

Disc_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Dis_logit_real,labels=y_real))
Disc_loss_gen = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Dis_logit_gen, labels=y_gen))

ent_real_loss = -tf.reduce_mean(
                        tf.reduce_sum(
                            tf.multiply(Dis_prob_real, tf.log(Dis_prob_real)), 1
                        )
                    )

ent_gen_loss = -tf.reduce_mean(
                        tf.reduce_sum(
                            tf.multiply(Dis_prob_gen, tf.log(Dis_prob_gen)), 1
                        )
                    )

Dis_loss = Disc_loss_real + Disc_loss_gen + 1.85 * ent_real_loss

# generator loss
pt_loss = pull_away_term(Dis_h2_tar_gen)

y_tar= tf.placeholder(tf.int32, shape=[None, Dis_dim[3]])
T_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Dis_logit_tar, labels=y_tar))
tar_thrld = tf.divide(tf.reduce_max(Dis_prob_tar_gen[:,-1]) +
                      tf.reduce_min(Dis_prob_tar_gen[:,-1]), 2)

indicator = tf.sign(
              tf.subtract(Dis_prob_tar_gen[:,-1],
                          tar_thrld))
condition = tf.greater(tf.zeros_like(indicator), indicator)
mask_tar = tf.where(condition, tf.zeros_like(indicator), indicator)
Gen_ent_loss = tf.reduce_mean(tf.multiply(tf.log(Dis_prob_tar_gen[:,-1]), mask_tar))

fm_loss = tf.reduce_mean(
            tf.sqrt(
                tf.reduce_sum(
                    tf.square(Dis_logit_real - Dis_logit_gen), 1
                    )
                )
            )

Gen_loss = pt_loss + Gen_ent_loss + fm_loss

Disc_solver = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(Dis_loss, var_list=parameters_Dis)
Gen_solver = tf.train.AdamOptimizer().minimize(Gen_loss, var_list=parameters_Gen)
T_solver = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(T_loss, var_list=parameters_T)

# Loading data

min_max_scaler = MinMaxScaler()


x_benign = min_max_scaler.fit_transform(np.load("./data/creditCard/ben_hid_repre_r2.npy"))
x_vandal = min_max_scaler.transform(np.load("./data/creditCard/van_hid_repre_r2.npy"))

x_benign = sample_shuffle_uspv(x_benign)
x_vandal = sample_shuffle_uspv(x_vandal)


x_pre = x_benign[0:700]

y_pre = np.zeros(len(x_pre))
y_pre = one_hot_encoder(y_pre, 2)

x_train = x_pre

y_real_mb = one_hot_encoder(np.zeros(minibatch_size), 2)
y_fake_mb = one_hot_encoder(np.ones(minibatch_size), 2)


x_test = x_benign[-490:].tolist() + x_vandal[-490:].tolist()
x_test = np.array(x_test)


y_test = np.zeros(len(x_test))

y_test[490:] = 1


sess = tf.Session()
sess.run(tf.global_variables_initializer())

# pre-training for target distribution

_ = sess.run(T_solver,
             feed_dict={
                X_tar:x_pre,
                y_tar:y_pre
                })

q = np.divide(len(x_train), minibatch_size)

# n_epoch = 1, while n_epoch:

d_ben_pro, d_fake_pro, fm_loss_coll = list(), list(), list()
f1_score  = list()
d_val_pro = list()

n_round = 200

for n_epoch in range(n_round):

    X_mb_oc = sample_shuffle_uspv(x_train)

    for n_batch in range(int(q)):

        _, D_loss_curr, ent_real_curr = sess.run([Disc_solver, Dis_loss, ent_real_loss],
                                          feed_dict={
                                                     X_oc: X_mb_oc[n_batch*minibatch_size:(n_batch+1)*minibatch_size],
                                                     Z: sampleY(minibatch_size, noise_dim),
                                                     y_real: y_real_mb,
                                                     y_gen: y_fake_mb
                                                     })

        _, G_loss_curr, fm_loss_curr = sess.run([Gen_solver, Gen_loss, fm_loss],
                                           feed_dict={Z: sampleY(minibatch_size, noise_dim),
                                                      X_oc: X_mb_oc[n_batch*minibatch_size:(n_batch+1)*minibatch_size],
                                                      })

    Dis_prob_real_, Dis_prob_gen_ = sess.run([Dis_prob_real, Dis_prob_gen],
                                         feed_dict={X_oc: x_train,
                                                    Z: sampleY(len(x_train), noise_dim)})

    
    Dis_prob_vandal_ = sess.run(Dis_prob_real,
                                feed_dict={X_oc:x_vandal[-490:]})

    d_ben_pro.append(np.mean(Dis_prob_real_[:, 0]))
    d_fake_pro.append(np.mean(Dis_prob_gen_[:, 0]))
    d_val_pro.append(np.mean(Dis_prob_vandal_[:, 0]))
    fm_loss_coll.append(fm_loss_curr)

    prob, _ = sess.run([Dis_prob_real, Dis_logit_real], feed_dict={X_oc: x_test})
    y_prediction = np.argmax(prob, axis=1)
    conf_mat = classification_report(y_test, y_prediction, target_names=['benign', 'vandal'], digits=4)
    f1_score.append(float(list(filter(None, conf_mat.strip().split(" ")))[12]))

if not dra_tra_pro:
    acc = np.sum(y_prediction == y_test)/float(len(y_prediction))
    print (conf_mat)
    print ("accurancy:%s"%acc)

if dra_tra_pro:
    draw_trends(fm_loss_coll, f1_score)

exit(0)
