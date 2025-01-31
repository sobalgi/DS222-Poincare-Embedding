import random, numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
from utils import pplot, PoincareBase, ranking
from tqdm import tqdm
import time

tf.logging.set_verbosity(tf.logging.INFO)

class PoincareTensor(PoincareBase):

    eps = 1e-6

    def __init__(self,num_iter=100,num_negs=10,lr1=0.001,lr2=0.00001,dp='/home/sourabh/prj/pe/wordnet/mammal_closure.tsv'): # dim=2
        super(PoincareTensor,self).__init__(num_iter,num_negs,lr1,lr2,dp)
    def proj(self,x):
        return tf.clip_by_norm(x,1.-self.eps,axes=1)

    def dists(self, u, v):
        uu, uv, vv = tf.norm(u) ** 2, tf.matmul(u, tf.transpose(v)), tf.norm(v) ** 2
        alpha, beta = tf.maximum(1 - uu, self.eps), tf.maximum(1 - vv, self.eps)
        gamma = tf.maximum(1 + 2 * (uu + vv - 2 * uv) / alpha / beta, 1 + self.eps)
        return tf.acosh(gamma)

    def grad(self, x, v):
        uu, uv, vv = tf.norm(x) ** 2, tf.matmul(x, tf.transpose(v)), tf.norm(v) ** 2
        alpha = 1-uu
        beta = 1-vv
        z = 1+ (2*(uu+vv-2*uv) / (alpha * beta))
        a = (vv - 2 * uv + 1) / tf.pow(alpha, 2)
        a = a * x - v / alpha
        z = tf.math.sqrt(tf.pow(z, 2) - 1)
        z = tf.maximum(z * beta, self.eps)
        return 4 * a / z

    def train(self):
        ld = len(self.pdata); lp = range(len(self.pdict))
        graph = tf.Graph()
        # with graph.as_default():
        # step = tf.Variable(0, trainable=False)
        step = tf.train.get_or_create_global_step()
        pembs = tf.Variable(tf.random_uniform([len(self.pdict), self.dim], minval=-0.001, maxval=0.001))
        n1 = tf.placeholder(tf.int32, shape=(1,), name='n1')
        n2 = tf.placeholder(tf.int32, shape=(1,), name='n2')
        sp = tf.placeholder(tf.int32, shape=(self.num_negs,), name='sp')
        u, v, negs = map(lambda x: tf.nn.embedding_lookup(pembs, x), [n1, n2, sp])
        loss = -tf.log(tf.exp(-self.dists(u, v)) / tf.reduce_sum(tf.exp(-self.dists(u, negs))))
        learning_rate = tf.train.polynomial_decay(self.lr1, step, self.num_iter * ld, self.lr2)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate * hvd.size())

        # Horovod: add Horovod Distributed Optimizer.
        optimizer = hvd.DistributedOptimizer(optimizer)
        hooks = [
            # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
            # from rank 0 to all other processes. This is necessary to ensure consistent
            # initialization of all workers when training is started with random weights
            # or restored from a checkpoint.
            hvd.BroadcastGlobalVariablesHook(0),

            # Horovod: adjust number of steps based on number of GPUs.
            tf.train.StopAtStepHook(last_step=self.num_iter* 13071 // hvd.size()),

            tf.train.LoggingTensorHook(tensors={'step': step, 'loss': loss},
                                       every_n_iter=1000),
        ]

        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())

        grad_vars = optimizer.compute_gradients(loss)
        rescaled = [(g * (1. - tf.reshape(tf.norm(v, axis=1), (-1, 1)) ** 2) ** 2 / 4., v) for g, v in grad_vars]
        trainstep = optimizer.apply_gradients(rescaled, global_step=step)
        pembs = self.proj(pembs)



        with tf.train.MonitoredTrainingSession(config=config, hooks=hooks) as session:
            while not session.should_stop():
                start = time.time()
                # init.run()

                print('HVD size: {0} '.format(hvd.size()))
                for epoch in range(self.num_iter):
                    total_loss = 0
                    #print epoch; random.shuffle(self.pdata)
                    total_loss = 0
                    for w1, w2 in self.pdata:
                        i1, i2 = self.pdict[w1], self.pdict[w2]
                        _, self.pembs, l = session.run([trainstep, pembs, loss], feed_dict= \
                            {n1: [i1], n2: [i2], sp: [random.choice(lp) for _ in range(self.num_negs)]})
                        total_loss += int(l[0][0])
                    print('Average loss epoch {0}: {1}'.format(epoch, total_loss))
                    # pplot(self.pdict, self.pembs, epoch, 'mammal_tensor')
                print('Total time: {0} seconds'.format(time.time() - start))



if __name__ == '__main__':
    # Horovod: initialize Horovod.
    hvd.init()

    PoincareTensor(num_iter=100).train()