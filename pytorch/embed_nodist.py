#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch as th
import numpy as np
import logging
import argparse
from torch.autograd import Variable
from collections import defaultdict as ddict
import torch.multiprocessing as mp
import model, train, rsgd
from data import slurp
from rsgd import RiemannianSGD
from sklearn.metrics import average_precision_score
import gc
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def ranking(types, model, distfn):
    lt = th.from_numpy(model.embedding())
    embedding = Variable(lt, volatile=True)
    ranks = []
    ap_scores = []
    for s, s_types in types.items():
        s_e = Variable(lt[s].expand_as(embedding), volatile=True)
        _dists = model.dist()(s_e, embedding).data.cpu().numpy().flatten()
        _dists[s] = 1e+12
        _labels = np.zeros(embedding.size(0))
        _dists_masked = _dists.copy()
        _ranks = []
        for o in s_types:
            _dists_masked[o] = np.Inf
            _labels[o] = 1
        ap_scores.append(average_precision_score(_labels, -_dists))
        for o in s_types:
            d = _dists_masked.copy()
            d[o] = _dists[o]
            r = np.argsort(d)
            _ranks.append(np.where(r == o)[0][0] + 1)
        ranks += _ranks
    # return np.mean(ranks), np.mean(ap_scores)  # TODO : SB
    return ranks, ap_scores  # TODO : SB


def control(queue, log, types, data, fout, distfn, nepochs, processes):
    min_rank = (np.Inf, -1)
    max_map = (0, -1)
    while True:
        gc.collect()
        msg = queue.get()
        if msg is None:
            for p in processes:
                p.terminate()
            break
        else:
            epoch, elapsed, loss, model = msg
        if model is not None:
            # save model to fout
            th.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'objects': data.objects,
            }, fout)
            # compute embedding quality
            # mrank, mAP = ranking(types, model, distfn)  # TODO : SB
            ranks, ap_scores = ranking(types, model, distfn)
            mrank, mAP = np.mean(ranks), np.mean(ap_scores)
            if mrank < min_rank[0]:
                min_rank = (mrank, epoch)
            if mAP > max_map[0]:
                max_map = (mAP, epoch)
            log.info(
                ('eval: {'
                 '"epoch": %d, '
                 '"elapsed": %.2f, '
                 '"loss": %.3f, '
                 '"mean_rank": %.2f, '
                 '"mAP": %.4f, '
                 '"best_rank": %.2f, '
                 '"best_mAP": %.4f}') % (
                    epoch, elapsed, loss, mrank, mAP, min_rank[0], max_map[0])
            )

            if model.dim == 2:
                plot_emb(types, model, epoch, data.objects, ap_scores, loss, min_rank[0], max_map[0])
        else:
            log.info(f'json_log: {{"epoch": {epoch}, "loss": {loss}, "elapsed": {elapsed}}}')
        if epoch >= nepochs - 1:
            log.info(
                ('results: {'
                 '"mAP": %g, '
                 '"mAP epoch": %d, '
                 '"mean rank": %g, '
                 '"mean rank epoch": %d'
                 '}') % (
                    max_map[0], max_map[1], min_rank[0], min_rank[1])
            )
            break
    if model.dim == 2:
        plot_emb(types, model, epoch, data.objects, ap_scores, loss, min_rank[0], max_map[0])


def plot_emb(types, model, epoch, objects, ap_scores, loss, best_rank, best_mAP):  #  TODO : SB
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.cla()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))
    ax.add_artist(plt.Circle((0, 0), 1., color='lightgrey', fill=False))

    emb = model.embedding()
    # visible_ents = np.unique(np.random.randint(low=0, high=len(ap_scores)-1, size=(int(.05*len(ap_scores)))))
    # visible_ents = [th.LongTensor([5])]
    printed_ents = []
    prob_u = np.random.uniform(0, 1, len(ap_scores))
    i = 0
    for s, s_types in tqdm(types.items()):
        # if s in visible_ents and s not in printed_ents:
        #     printed_ents.append(s)
        if prob_u[i] < .01:
            plt.plot(emb[s][0], emb[s][1], marker='b', color='black')  # plot the dot
            ax.text(emb[s][0] + .01, emb[s][1] + .01, objects[s].split('.')[0], color='g',
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))  # add text with box  dodgerblue
        # m = 0
        for o in s_types:
        #     if o not in visible_ents and m <= 4:
                # visible_ents.append(o)
                plt.plot([emb[s][0], emb[o][0]], [emb[s][1], emb[o][1]], c='cornflowerblue', linewidth=ap_scores[i] / max(ap_scores) * 5)
                # m += 1

        i += 1
    plt.text(-1, 1, 'epoch = ' + str(epoch+1) + '\nloss = ' + str(loss)[:5], bbox=dict(facecolor='red', alpha=0.5, edgecolor='black', boxstyle='round,pad=1'))
    plt.text(-1, -1, 'best_rank = ' + str(best_rank)[:5] + '\nbest_mAP = ' + str(best_mAP)[:5], bbox=dict(facecolor='red', alpha=0.5, edgecolor='black', boxstyle='round,pad=1'))
    plt.show(block=False)
    fig.savefig(model.model_name + '_e' + str(epoch + 1) + '.png', dpi=fig.dpi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Poincare Embeddings')
    parser.add_argument('-dim', help='Embedding dimension', type=int)
    parser.add_argument('-dset', help='Dataset to embed', type=str)
    parser.add_argument('-fout', help='Filename where to store model', type=str)
    parser.add_argument('-distfn', help='Distance function', type=str)
    parser.add_argument('-lr', help='Learning rate', type=float)
    parser.add_argument('-epochs', help='Number of epochs', type=int, default=200)
    parser.add_argument('-batchsize', help='Batchsize', type=int, default=50)
    parser.add_argument('-negs', help='Number of negatives', type=int, default=20)
    parser.add_argument('-nproc', help='Number of processes', type=int, default=5)
    parser.add_argument('-ndproc', help='Number of data loading processes', type=int, default=2)
    parser.add_argument('-eval_each', help='Run evaluation each n-th epoch', type=int, default=10)
    parser.add_argument('-burnin', help='Duration of burn in', type=int, default=20)
    parser.add_argument('-debug', help='Print debug output', action='store_true', default=False)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
    # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # parser.add_argument('--fp16-allreduce', action='store_true', default=False,
    #                     help='use fp16 compression during allreduce')
    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda and th.cuda.is_available()

    th.manual_seed(opt.seed)

    th.set_default_tensor_type('torch.FloatTensor')
    if opt.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    log = logging.getLogger('poincare-nips17')
    logging.basicConfig(level=log_level, format='%(message)s', stream=sys.stdout)

    print('create dictionary of entities \n')
    idx, objects = slurp(opt.dset)

    # create adjacency list for evaluation
    print('create adjacency list for evaluation\n')
    adjacency = ddict(set)  # { tail : head}
    r_adjacency = ddict(set)  # { head : tail }
    for i in tqdm(range(len(idx))):
        s, o, _ = idx[i]
        # adjacency[int(s)].add(int(o))  # TODO : SB
        adjacency[s].add(o)
        r_adjacency[s].add(o)
    adjacency = dict(adjacency)

    # setup Riemannian gradients for distances
    opt.retraction = rsgd.euclidean_retraction
    if opt.distfn == 'poincare':
        distfn = model.PoincareDistance
        opt.rgrad = rsgd.poincare_grad
    elif opt.distfn == 'euclidean':
        distfn = model.EuclideanDistance
        opt.rgrad = rsgd.euclidean_grad
    elif opt.distfn == 'transe':
        distfn = model.TranseDistance
        opt.rgrad = rsgd.euclidean_grad
    else:
        raise ValueError(f'Unknown distance function {opt.distfn}')

    # initialize model and data
    model, data, model_name, conf = model.SNGraphDataset.initialize(distfn, opt, idx, objects)

    # Build config string for log
    conf = [
               ('distfn', '"{:s}"'),
               ('dim', '{:d}'),
               ('lr', '{:g}'),
               ('batchsize', '{:d}'),
               ('negs', '{:d}'),
           ] + conf
    conf = ', '.join(['"{}": {}'.format(k, f).format(getattr(opt, k)) for k, f in conf])

    log.info(f'json_conf: {{{conf}}}')

    # initialize optimizer
    optimizer = RiemannianSGD(
        model.parameters(),
        rgrad=opt.rgrad,
        retraction=opt.retraction,
        lr=opt.lr,
    )

    # if nproc == 0, run single threaded, otherwise run Hogwild
    if opt.nproc == 0:
        train.train(model, data, optimizer, opt, log, 0)
    else:
        queue = mp.Manager().Queue()
        model.share_memory()
        processes = []
        print('create process for parallel processing\n')
        for rank in range(opt.nproc):
            p = mp.Process(
                target=train.train_mp,
                args=(model, data, optimizer, opt, log, rank + 1, queue)
            )
            p.start()
            processes.append(p)

        ctrl = mp.Process(
            target=control,
            args=(queue, log, adjacency, data, opt.fout, distfn, opt.epochs, processes)
        )
        ctrl.start()
        ctrl.join()
