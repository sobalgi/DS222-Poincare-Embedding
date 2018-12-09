import operator
import matplotlib.pyplot as plt
from collections import defaultdict
from nltk.corpus import wordnet as wn
from functools import reduce
import numpy as np
import math
from sklearn.metrics import average_precision_score

plt.style.use('ggplot')

eps = 1e-6
def gen_data(network=defaultdict(set)):
    words, target = wn.words(), wn.synset('mammal.n.01')
    targets = set(open('/home/sourabh/prj/petf/targets.txt').read().split('\n'))
    nouns = {noun for word in words for noun in wn.synsets(word, pos='n') if noun.name() in targets}
    for noun in nouns:
        for path in noun.hypernym_paths():
            if not target in path: continue
            for i in range(path.index(target), len(path) - 1):
                if not path[i].name() in targets: continue
                network[noun.name()].add(path[i].name())
    with open('/home/sourabh/prj/pe/wordnet/mammal_closure.tsv', 'w') as out:
        for key, vals in network.items():
            for val in vals: out.write(key + '\t' + val + '\n')

def dists(u,v):
    uu, uv, vv = np.linalg.norm(u)**2, np.matmul(u,v), np.linalg.norm(v)**2
    alpha, beta = np.maximum(1-uu,eps), np.maximum(1-vv,eps)
    z = 1+(2*(uu+vv-2*uv)/(alpha*beta))
    gamma = np.maximum(z,1+eps)
    return np.arccosh(gamma)

def ranking(types, embedding):
    # lt = th.from_numpy(model.embedding())
    # embedding = Variable(lt, volatile=True)
    ranks = []
    ap_scores = []

    for s, s_types in types.items():
        _dists = np.zeros(embedding.shape[0])
        for i in range(embedding.shape[0]):
            _dists[i] = dists(embedding[s],embedding[i,:])
        _dists[s] = 1e+12

        _labels = np.zeros(embedding.shape[0])
        _dists_masked = np.copy(_dists)
        _ranks = []

        for o in s_types:
            _dists_masked[o] = np.Inf
            _labels[o] = 1

        #~np.isfinite(average_precision_score(_labels, -_dists))
        ap_scores.append(average_precision_score(_labels, -_dists))
        ap_scores = [0 if ~np.isfinite(x) else x for x in ap_scores]
        #print(average_precision_score(_labels, -_dists), np.where(~np.isfinite(-_dists)), _labels)

        for o in s_types:
            d = np.copy(_dists_masked)
            d[o] = _dists[o]
            r = np.argsort(d)
            _ranks.append(np.where(r == o)[0][0] + 1)
        ranks += _ranks
    return np.mean(ranks), np.mean(ap_scores)


def pplot(pdict, p, epoch, name='mammal'):
    fig = plt.figure(figsize=(20, 20))
    ax = plt.gca()
    ax.cla()
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))
    ax.add_artist(plt.Circle((0, 0), 1., color='black', fill=False))
    for w, i in pdict.items():
        c0, c1 = p[i]
        ax.plot(c0, c1, 'o', color='y')
        ax.text(c0 + .01, c1 + .01, w, color='b')
    fig.savefig('/home/rishixtreme/PycharmProjects/poincare/data/' + name + str(epoch) + '.png', dpi=fig.dpi)


class PoincareBase(object):
    def __init__(self, num_iter=140, num_negs=10, lr1=0.0001,lr2=0.00001,
                 dp='/home/sourabh/prj/pe/wordnet/mammal_closure.tsv'):  # dim=2
        self.dim = 2
        self.num_iter = num_iter

        self.num_negs = num_negs
        self.lr1, self.lr2 = lr1, lr2
        self.eps = 1e-6
        with open(dp) as f:
            # input = f.read().splitlines()
            a = set(f.read().split())

        with open(dp) as f:
            input = f.read().splitlines()
        # a = set(f.read().split())
        self.pdata = list(map(lambda l: l.split('\t'), input))

        self.pdict = {w: int(i) for i, w in enumerate(a)}

        self.mapdict = {}
        for k,val in self.pdict.items():
            a = []
            for i in range(len(self.pdata)):
                if self.pdata[i][0] == k:
                    a += [self.pdict[self.pdata[i][1]]]
            self.mapdict[val] = a
        #print(self.mapdict)



# print(self.pdict)

# self.pdict = {w:i for i,w in enumerate(set(reduce(operator.add,list(self.pdata))))}


# def dists(self,u,v): pass
# def train(self): pass

if __name__ == '__main__':
    gen_data()
