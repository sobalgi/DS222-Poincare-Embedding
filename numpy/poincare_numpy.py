import nltk
from nltk.corpus import wordnet as wn
from math import *
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

STABILITY = 0.00001 # to avoid overflow while dividing
network = {} # representation of network (here it is hierarchical)

last_level = 8

# plots the embedding of all the nodes of network
def plotall(ii):
    fig = plt.figure()

    # plot all the nodes
    for a in emb:
        z = "{0:03b}".format(levelOfNode[a])
        z = np.array(list(z)).astype(int)
        plt.plot(emb[a][0], emb[a][1], marker = 'o', color = z)
    # plot the relationship, black line means root level relationship
    # consecutive relationship lines fade out in color
    for a in network:
        z = "{0:03b}".format(levelOfNode[a])
        z = np.array(list(z)).astype(int)
        for b in network[a]:
            plt.plot([emb[a][0], emb[b][0]], [emb[a][1], emb[b][1]], color = z)
    # plt.show()
    fig.savefig("adam" + str(last_level) + '_' + str(ii) + '.png', dpi=fig.dpi)

# network: the actual network of which node is connected to whom
# levelOfNode: level of the node in hierarchical data
levelOfNode = {}

# recursive function to populate the hyponyms of a root node in `network`
# synset: the root node
# last_level: the level till which we consider the hyponyms
def get_hyponyms(synset, level):
    if (level == last_level):
        levelOfNode[str(synset)] = level
        return
    # BFS
    if not str(synset) in network:
        network[str(synset)] = [str(s) for s in synset.hyponyms()]
        levelOfNode[str(synset)] = level
    for hyponym in synset.hyponyms():
        get_hyponyms(hyponym, level + 1)

mammal = wn.synset('mammal.n.01')
get_hyponyms(mammal, 0)
levelOfNode[str(mammal)] = 0

# embedding of nodes of network
emb = {}

# Randomly uniform distribution
for a in network:
    for b in network[a]:
        emb[b] = np.random.uniform(low=-0.001, high=0.001, size=(2,))
    emb[a] = np.random.uniform(low=-0.001, high=0.001, size=(2,))

vocab = list(emb.keys())
random.shuffle(vocab)

# the leave nodes are not connected to anything
for a in emb:
    if not a in network:
        network[a] = []


# Partial derivative as given in the paper wrt theta
def partial_der(theta, x, gamma): #eqn4
    alpha = (1.0-np.dot(theta, theta))
    norm_x = np.dot(x, x)
    beta = (1-norm_x)
    gamma = gamma
    return 4.0/(beta * sqrt(gamma*gamma - 1) + STABILITY)*((norm_x- 2*np.dot(theta, x)+1)/(pow(alpha,2)+STABILITY)*theta - x/(alpha + STABILITY))

# a little modified update equation for adam
def update(emb, error_): #eqn5
    try:
        update =  pow((1 - np.dot(emb,emb)), 2)*error_/4
        # print (update)
        emb = emb - update
        if (np.dot(emb, emb) >= 1):
            emb = emb/sqrt(np.dot(emb, emb)) - 0.00001
        return emb
    except Exception as e:
        print (e)
        temp = input()

# Distance in poincare disk model
def dist(vec1, vec2): # eqn1
    return 1 + 2*np.dot(vec1 - vec2, vec1 - vec2)/ \
             ((1-np.dot(vec1, vec1))*(1-np.dot(vec2, vec2)) + STABILITY)



num_negs = 5

plotall("init")

# adam variables as per adam paper
update_emb = emb.copy()

m = {}
v = {}
t = {}
for a in emb:
    m[a] = np.asarray([0.0, 0.0])
    v[a] = np.asarray([0.0, 0.0])

eps = 0.00000001
b1 = 0.9
b2 = 0.999
b1_t = 0.9
b2_t = 0.999
alpha = 0.005


for epoch in tqdm(range(20001)):
    # update_emb is the update we would like to make to an embedding
    for i in update_emb:
        update_emb[i] = np.zeros(2)
    # pos2 is related to pos1
    # negs are not related to pos1
    for pos1 in vocab:
        if not network[pos1]: # a leaf node
            continue
        pos2 = random.choice(network[pos1]) # pos2 and pos1 are related
        dist_p_init = dist(emb[pos1], emb[pos2]) # distance between the related nodes
        if (dist_p_init > 700): # this causes overflow, so I clipped it here
            print ("got one very high") # if you have reached this zone, the training is unstable now
            dist_p_init = 700
        elif (dist_p_init < -700):
            print ("got one very high")
            dist_p_init = -700
        dist_p = cosh(dist_p_init) # this is the actual distance, it is always positive
        # print ("distance between related nodes", dist_p)
        negs = [] # pairs of not related nodes, the first node in the pair is `pos1`
        dist_negs_init = [] # distances without taking cosh on it (for not related nodes)
        dist_negs = [] # distances with taking cosh on it (for not related nodes)
        while (len(negs) < num_negs):
            neg1 = pos1
            neg2 = random.choice(vocab)
            if not (neg2 in network[neg1] or neg1 in network[neg2] or neg2 == neg1): # neg2 should not be related to neg1 and vice versa
                dist_neg_init = dist(emb[neg1], emb[neg2])
                if (dist_neg_init > 700 or dist_neg_init < -700): # already dist is good, leave it
                    continue
                negs.append([neg1, neg2])
                dist_neg = cosh(dist_neg_init)
                dist_negs_init.append(dist_neg_init) # saving it for faster computation
                dist_negs.append(dist_neg)
                # print ("distance between non related nodes", dist_neg)
        loss_den = 0.0
        # eqn6
        for dist_neg in dist_negs:
            loss_den += exp(-1*dist_neg)
        loss = -1*dist_p - log(loss_den + STABILITY)
        # derivative of loss wrt positive relation [d(u, v)]
        der_p = -1
        der_negs = []
        # derivative of loss wrt negative relation [d(u, v')]
        for dist_neg in dist_negs:
            der_negs.append(exp(-1*dist_neg)/(loss_den + STABILITY))
        # derivative of loss wrt pos1
        der_p_pos1 = der_p * partial_der(emb[pos1], emb[pos2], dist_p_init)
        # derivative of loss wrt pos2
        der_p_pos2 = der_p * partial_der(emb[pos2], emb[pos1], dist_p_init)
        der_negs_final = []
        for (der_neg, neg, dist_neg_init) in zip(der_negs, negs, dist_negs_init):
            # derivative of loss wrt second element of the pair in neg
            der_neg1 = der_neg * partial_der(emb[neg[1]], emb[neg[0]], dist_neg_init)
            # derivative of loss wrt first element of the pair in neg
            der_neg0 = der_neg * partial_der(emb[neg[0]], emb[neg[1]], dist_neg_init)
            der_negs_final.append([der_neg0, der_neg1])
        # update embeddings
        update_emb[pos1] -= der_p_pos1
        update_emb[pos2] -= der_p_pos2
        for (neg, der_neg) in zip(negs, der_negs_final):
            update_emb[neg[0]] -= der_neg[0]
            update_emb[neg[1]] -= der_neg[1]
    # adam update now
    b1_t *= b1
    b2_t *= b2
    for i in update_emb:
        m[i] = b1*m[i] + (1.0 - b1) * update_emb[i]
        v[i] = b2*v[i] + (1.0 - b2) * update_emb[i] * update_emb[i]
        update_emb[i] =  alpha * (m[i] / (1.0 - b1_t)) / np.sqrt((v[i] / (1.0 - b2_t)) + eps);
        emb[i] = update(emb[i], update_emb[i])
    # plot the embeddings
    if ((epoch)%100 == 0):
        plotall(epoch+1)

