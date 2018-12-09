#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch as th
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.optim import Adam
import horovod.torch as hvd
import numpy as np
import logging
import argparse
from torch.autograd import Variable
from collections import defaultdict as ddict
import model, train, rsgd
from data import slurp
from rsgd import RiemannianSGD
from sklearn.metrics import average_precision_score
import sys
from tqdm import tqdm
import timeit
import pickle as pkl
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


def plot_emb(types, model, epoch, objects, ap_scores, loss, best_rank, best_mAP):  #  TODO : SB
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.cla()
#     fig.patch.set_visible(False)
    ax.axis('off')
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))
    ax.add_artist(plt.Circle((0, 0), 1., color='lightgrey', fill=True))

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
            plt.plot(emb[s][0], emb[s][1], marker='o', color='black')  # plot the dot
            ax.text(emb[s][0] + .01, emb[s][1] + .01, objects[s].split('.')[0], color='orange',
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))  # add text with box  dodgerblue
        # m = 0
        for o in s_types:
        #     if o not in visible_ents and m <= 4:
                # visible_ents.append(o)
                plt.plot([emb[s][0], emb[o][0]], [emb[s][1], emb[o][1]], c='cornflowerblue', linewidth=ap_scores[i] / max(ap_scores) * 5)
                # m += 1

        i += 1
    plt.text(-1, 1, 'epoch = ' + str(epoch+1) + '\nloss = ' + str(loss.item())[:5], bbox=dict(facecolor='red', alpha=0.5, edgecolor='black', boxstyle='round,pad=1'))
    plt.text(-1, -1, 'best_rank = ' + str(best_rank)[:5] + '\nbest_mAP = ' + str(best_mAP)[:5], bbox=dict(facecolor='red', alpha=0.5, edgecolor='black', boxstyle='round,pad=1'))
    # plt.show(block=False)
    fig.savefig(model.model_name + '_e' + str(epoch + 1) + '.png', dpi=fig.dpi)


def plot_progress(file_name='logs.pkl'):
    # fig = plt.figure()

    logs = pkl.load(open(file_name, 'rb'))

    # (epoch, avg_epoch_loss, total_epoch_time, mrank, mAP)

    epoch = [e[0] for e in logs]
    avg_epoch_loss = [e[1] for e in logs]
    total_epoch_time = [e[2] for e in logs]
    mrank = [e[3] for e in logs]
    mAP = [e[4] for e in logs]

    plt.subplot(2, 2, 1)
    plt.plot(epoch, avg_epoch_loss, 'o-')
    plt.title('Loss v/s Epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(2, 2, 2)
    plt.plot(total_epoch_time, avg_epoch_loss, 'o-')
    plt.title('Loss v/s Time')
    plt.xlabel('time')
    plt.ylabel('loss')

    plt.subplot(2, 2, 3)
    plt.plot(epoch, mrank, 'o-')
    plt.title('Mean Rank v/s Epoch')
    plt.xlabel('epoch')
    plt.ylabel('rank')

    plt.subplot(2, 2, 4)
    plt.plot(epoch, mAP, 'o-')
    plt.title('Mean Average Precision v/s Epoch')
    plt.xlabel('epoch')
    plt.ylabel('MAP')

    # plt.show(block=False)
    # plt.ion()
    # plt.show()
    plt.savefig(file_name + '.png')


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

    # Initialize Horovod
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    th.cuda.set_device(hvd.local_rank())

    th.set_default_tensor_type('torch.FloatTensor')
    if opt.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    log = logging.getLogger('poincare-nips17')
    logging.basicConfig(level=log_level, format='%(message)s', stream=sys.stdout)

    print('create dictionary of entities \n')
    idx, objects = slurp(opt.dset)

    'mammal.n.01'

    # create adjacency list for evaluation
    print('create adjacency list for evaluation\n')
    adjacency = ddict(set)  # { tail : head}
    # r_adjacency = ddict(set)  # { head : tail }
    for i in tqdm(range(len(idx))):
        s, o, _ = idx[i]
        # adjacency[int(s)].add(int(o))  # TODO : SB
        adjacency[s].add(o)
        # r_adjacency[s].add(o)
    adjacency = dict(adjacency)
    # r_adjacency = dict(r_adjacency)

    # 'mammal.n.01'

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

    if opt.cuda:
        model.cuda()

    # Partition dataset among workers using DistributedSampler
    train_sampler = DistributedSampler(
        data, num_replicas=hvd.size(), rank=hvd.rank())

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
        lr=opt.lr * hvd.size(),
    )

    print(f'Size of hvd process : {hvd.size()}')

    # optimizer = Adam(
    #     model.parameters(),
    #     lr=opt.lr * hvd.size(),
    # )

    lr = opt.lr

    # Add Horovod Distributed Optimizer
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    _lr_multiplier = 0.1
    NoDisplayObj = 66  # Number of entities to display on the graph

    # scheduler = MultiStepLR(optimizer, milestones=[opt.burnin]+list(range(int(opt.epochs/10), opt.epochs, int(opt.epochs/10))), gamma=_lr_multiplier)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

    # Broadcast parameters from rank 0 to all other processes.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    loader = th.utils.data.DataLoader(
        data,
        batch_size=opt.batchsize,
        #shuffle=True,
        num_workers=opt.ndproc,
        collate_fn=data.collate,
        sampler=train_sampler
    )

    log.info(f'Burnin: lr={opt.lr * _lr_multiplier}')

    obj_id = np.random.randint(0, len(data.objects), size=NoDisplayObj)
    obj_name = [data.objects[i].split('.')[0] for i in obj_id]

    min_rank = (np.Inf, -1)
    max_map = (0, -1)

    logs = []

    total_epoch_time = 0
    epoch_loss = []
    epoch_time = []
    for epoch in range(opt.epochs):
        avg_epoch_loss_sum = 0
        steps = 0

        # scheduler.step()
        loss = None
        data.burnin = False
        t_start = timeit.default_timer()
        for inputs, targets in loader:
            elapsed = timeit.default_timer() - t_start
            optimizer.zero_grad()
            preds = model(inputs.cuda())
            loss = model.loss(preds, targets.cuda(), size_average=True)
            loss.backward()
            optimizer.step()

            steps += 1
            avg_epoch_loss_sum += loss.data[0].detach().cpu().numpy()

        avg_epoch_loss = float(avg_epoch_loss_sum) / steps
        epoch_loss.append(avg_epoch_loss)

        if hvd.rank() == 0:
            emb = None
            if epoch == (opt.epochs - 1) or epoch % opt.eval_each == (opt.eval_each - 1):
                emb = model

                if model is not None:
                    # save model to fout
                    th.save({
                        'model': model.state_dict(),
                        'epoch': epoch,
                        'objects': data.objects,
                    }, opt.fout)
                    # compute embedding quality
                    # mrank, mAP = ranking(types, model, distfn)  # TODO : SB
                    ranks, ap_scores = ranking(adjacency, model, distfn)
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

                    total_epoch_time += float(elapsed)

                    logs.append((epoch, avg_epoch_loss, total_epoch_time, mrank, mAP))
                    pkl.dump(logs,
                             open('logs.pkl', 'wb'))

                    plot_progress(file_name='logs.pkl')

                    if model.dim == 2:
                        plot_emb(adjacency, model, epoch, data.objects, ap_scores, loss, min_rank[0], max_map[0])
                else:
                    log.info(f'json_log: {{"epoch": {epoch}, "loss": {loss}, "elapsed": {elapsed}}}')
                if epoch >= opt.epochs - 1:
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
        log.info(
            'info: {'
            f'"epoch": {epoch}, '
            f'"rank": {hvd.rank()}, '
            f'"local_rank": {hvd.local_rank()}, '
            f'"elapsed": {elapsed}, '
            f'"loss": {np.mean(epoch_loss)}, '
            '}'
        )
        # pkl.dump((epoch, avg_epoch_loss, total_epoch_time, hvd.rank(), hvd.local_rank()), open('logs.pkl', 'wb'))

    if model.dim == 2:
        plot_emb(adjacency, model, epoch, data.objects, ap_scores, loss, min_rank[0], max_map[0])

    log.info(
        'info: {'
        f'"elapsed": {elapsed}, '
        f'"loss": {np.mean(epoch_loss)}, '
        '}'
    )

    if model is not None:
        # save model to fout
        th.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'objects': data.objects,
        }, opt.fout)
