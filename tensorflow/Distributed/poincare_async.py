import random, numpy as np
import tensorflow as tf
from utils import pplot, PoincareBase
from tqdm import tqdm
import time


def proj(x):
    return tf.clip_by_norm(x, 1. - p.eps, axes=1)


def dists(u, v):
    uu, uv, vv = tf.norm(u) ** 2, tf.matmul(u, tf.transpose(v)), tf.norm(v) ** 2
    alpha, beta = tf.maximum(1 - uu, p.eps), tf.maximum(1 - vv, p.eps)
    gamma = tf.maximum(1 + 2 * (uu + vv - 2 * uv) / alpha / beta, 1 + p.eps)
    return tf.acosh(gamma)

# cluster specification
parameter_servers = ["10.24.1.203:2220"]
workers = [ "10.24.1.207:2220",
            "10.24.1.208:2220"]
#			"10.24.1.210:2220",
#			"10.24.1.213:2220"]

cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(
    cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index)

if FLAGS.job_name == "ps":
    server.join()
    print('\n parameter servers initialized ...')
elif FLAGS.job_name == "worker":
    print('\n workers initialized ...')
    print("Data Loading ... \n")

    p = PoincareBase()

    num_iter=50
    num_negs=10
    lr1 = 1
    lr2 = 0.1
    #dp='/home/rishixtreme/PycharmProjects/poincare/data/mammal_subtree.tsv'): # dim=2
    ld = len(p.pdata); lp = range(len(p.pdict))
    print('ld',ld)
    print('lp',lp)

    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):

        global_step = tf.get_variable(
            'global_step',
            [], dtype=tf.int32,
            initializer=tf.constant_initializer(0),
            trainable=False)
        #graph = tf.Graph()
        print(num_iter)
        step = tf.Variable(0, trainable=False)
        pembs = tf.Variable(tf.random_uniform([len(p.pdict), p.dim], minval=-0.001, maxval=0.001))
        n1 = tf.placeholder(tf.int32, shape=(1,), name='n1')
        n2 = tf.placeholder(tf.int32, shape=(1,), name='n2')
        sp = tf.placeholder(tf.int32, shape=(p.num_negs,), name='sp')
        u, v, negs = map(lambda x: tf.nn.embedding_lookup(pembs, x), [n1, n2, sp])
        loss = -tf.log(tf.exp(dists(u, v)) / tf.reduce_sum(tf.exp(-dists(u, negs))))
        learning_rate = tf.train.polynomial_decay(lr1, step, p.num_iter * ld, lr2)
        #opt = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        #optimizer = Opt.minimize(loss, global_step=global_step)
        grad_vars = optimizer.compute_gradients(loss)
        rescaled = [(g * (1. - tf.reshape(tf.norm(v, axis=1), (-1, 1)) ** 2) ** 2 / 4., v) for g, v in grad_vars]
        trainstep = optimizer.apply_gradients(rescaled, global_step=step)
        pembs = proj(pembs)
        init = tf.global_variables_initializer()

sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                         global_step=step, init_op=init)

with sv.prepare_or_wait_for_session(server.target) as sess:
    start = time.time()
    for epoch in range(num_iter):
        #print(epoch)
        #random.shuffle(p.pdata)
        total_loss = 0
        for w1,w2 in p.pdata:
            i1,i2 = p.pdict[w1], p.pdict[w2]
            _,p.pembs,l = sess.run([trainstep,pembs,loss],feed_dict= \
                {n1:[i1],n2:[i2],sp:[random.choice(lp) for _ in range(p.num_negs)]})
            total_loss += int(l[0][0])
        print('Average loss epoch {0}: {1}'.format(epoch, total_loss))
        #pplot(p.pdict,p.pembs,epoch,'mammal_tensor')
    print('Total time: {0} seconds'.format(time.time() - start))
    sv.stop()
