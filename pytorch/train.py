#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import timeit
import torch as th
from torch.utils.data import DataLoader
import gc
from numpy.random import randint
import matplotlib.pyplot as plt
plt.style.use('ggplot')

_lr_multiplier = 0.01
NoDisplayObj = 66  # Number of entities to display on the graph

def train_mp(model, data, optimizer, opt, log, rank, queue):
    try:
        train(model, data, optimizer, opt, log, rank, queue)
    except Exception as err:
        log.exception(err)
        queue.put(None)


def train(model, data, optimizer, opt, log, rank=1, queue=None):
    # setup parallel data loader
    loader = DataLoader(
        data,
        batch_size=opt.batchsize,
        shuffle=True,
        num_workers=opt.ndproc,
        collate_fn=data.collate
    )
    obj_id = randint(0, len(data.objects), size=NoDisplayObj)
    obj_name = [data.objects[i].split('.')[0] for i in obj_id]

    for epoch in range(opt.epochs):
        epoch_loss = []
        loss = None
        data.burnin = False
        lr = opt.lr
        t_start = timeit.default_timer()
        if epoch < opt.burnin:
            data.burnin = True
            lr = opt.lr * _lr_multiplier
            if rank == 1:
                log.info(f'Burnin: lr={lr}')
        for inputs, targets in loader:
            elapsed = timeit.default_timer() - t_start
            optimizer.zero_grad()
            preds = model(inputs)
            loss = model.loss(preds, targets, size_average=True)
            loss.backward()
            optimizer.step(lr=lr)
            epoch_loss.append(loss.data[0])

        if rank == 1:
            emb = None
            if epoch == (opt.epochs - 1) or epoch % opt.eval_each == (opt.eval_each - 1):
                emb = model
                # plot_emb(emb, obj_id, obj_name, epoch)

            if queue is not None:
                queue.put(
                    (epoch, elapsed, np.mean(epoch_loss), emb)
                )
            else:
                log.info(
                    'info: {'
                    f'"elapsed": {elapsed}, '
                    f'"loss": {np.mean(epoch_loss)}, '
                    '}'
                )
        gc.collect()


# def plot_emb(types, model, epoch, objects, ap_scores):  #  TODO : SB
#     fig = plt.figure(figsize=(10, 10))
#     ax = plt.gca()
#     ax.cla()
#     # fig.patch.set_visible(False)
#     # ax.axis('off')
#     ax.set_xlim((-1.1, 1.1))
#     ax.set_ylim((-1.1, 1.1))
#     ax.add_artist(plt.Circle((0, 0), 1., color='lightgrey', fill=False))
#
#     emb = model.embedding()
#     i = 0
#     for s, s_types in types.items():
#         plt.plot(emb[s][0], emb[s][1], marker='o', color='black')  # plot the dot
#         if i < 100:
#             ax.text(emb[s][0] + .01, emb[s][1] + .01, objects[s].split('.')[0], color='dodgerblue',
#                 bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))  # add text with box
#         for o in s_types:
#             plt.plot([emb[s][0], emb[o][0]], [emb[s][1], emb[o][1]], c='cornflowerblue', linewidth=ap_scores[i] / max(ap_scores) * 5)
#         i += 1
#     fig.savefig(model.model_name + '_e' + str(epoch + 1) + '.png', dpi=fig.dpi)
