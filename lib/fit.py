from torch.autograd import Variable
from torch import nn
import torch

import os
import time
import numpy as np

def fit(model, loader, train2_loader, validate_loader, optimizer, save_path, cuda=True, loss='MSE'):
    if loss == 'MSE':
        # loss
        loss_fn = nn.MSELoss()
        if cuda:
            loss_fn = loss_fn.cuda()
    # setup metric recording
    loss_train_batch_history = []
    lr_batch_history = []
    loss_train_epoch_history = []
    loss_valid_epoch_history = []
    time_epoch_history = []
    epoch_list = []

    # setup initial loss_valid
    epoch = 1
    loss_valid_best = 100
    patience = 1
    num_epochs_increased = 0
    best_epoch = 1

    # Perform training
    while True:
        # Run one iteration of SGD
        t0 = time.time()
        loss_per_batch, lr_per_batch = train(model, loader, optimizer, cuda, loss_fn)
        # loss_train_batch_history = loss_train_batch_history + loss_per_batch
        loss_train_batch_history.append(loss_per_batch)
        lr_batch_history.append(lr_per_batch)

        # Estimate training and validation losses
        loss_train = loss_compute(model, train2_loader, loss_fn, cuda)
        loss_valid = loss_compute(model, validate_loader, loss_fn, cuda)
        loss_train_epoch_history.append(loss_train)
        loss_valid_epoch_history.append(loss_valid)
        diff_loss = (loss_valid - loss_valid_best) / loss_valid_best * 100
        time_epoch = time.time() - t0
        time_epoch_history.append(time_epoch)
        epoch_list.append(epoch)

        # display results
        # print('E: {:02d} / Train: {:.3e} / Valid: {:.3e} / Diff Valid: {:.2f}% / Diff Valid-Train: {:.1f}% / Time: {:.2f}'.format(epoch, loss_train, loss_valid, diff_loss, (loss_valid - loss_train)/loss_train*100, time_epoch))

        # if validation loss improves
        if diff_loss < 0:
            num_epochs_increased = 0

            # record epoch and loss
            best_epoch = epoch
            loss_valid_best = loss_valid

            # save the model
            torch.save(model.state_dict(), os.path.join(save_path, 'model.dat'))

            # save the other stuff
            np.savez(os.path.join(save_path, 'loss_results'),
                        loss_train_batch_history=loss_train_batch_history,
                        lr_batch_history=lr_batch_history,
                        loss_train_epoch_history=loss_train_epoch_history,
                        loss_valid_epoch_history=loss_valid_epoch_history,
                        time_epoch_history=time_epoch_history,
                        epoch_list=epoch_list,
                        best_epoch=epoch)
        else:
            num_epochs_increased += 1


        # stop training if we lose patience:
        if num_epochs_increased > patience:
            break

        # advance epoch counter
        epoch += 1


    # save the other stuff
    np.savez(os.path.join(save_path, 'loss_results'),
                loss_train_batch_history=loss_train_batch_history,
                lr_batch_history=lr_batch_history,
                loss_train_epoch_history=loss_train_epoch_history,
                loss_valid_epoch_history=loss_valid_epoch_history,
                time_epoch_history=time_epoch_history,
                epoch_list=epoch_list,
                best_epoch=epoch)

    diff_percent = (loss_valid - loss_train)/loss_train*100
    # print('Train: {:.3e} / Valid: {:.3e} / Diff Valid: {:.2f}% / Diff Valid-Train: {:.1f}%'.format(loss_train, loss_valid, diff_loss, diff_percent))
    return loss_train, diff_percent

def train(model, loader, optimizer, cuda, loss_fn):
    model.train()

    # keep track of loss per batch
    loss_per_batch = []
    lr_per_batch = []

    for batch_idx, data in enumerate(loader):
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        lr_per_batch.append(current_learning_rate)

        inputs = Variable(data[0], requires_grad=False)
        targets = Variable(data[1], requires_grad=False)
        if cuda == True:
            inputs = inputs.cuda()
            targets = targets.cuda(async=True) # CHANGED

        outputs = model(inputs)

        # print("fit.train: outputs.shape =", outputs.shape)

        model.zero_grad()

        loss = loss_fn(outputs, targets)

        loss_per_batch.append(loss.data[0])

        loss.backward()

        optimizer.step()

    return loss_per_batch, lr_per_batch

def loss_compute(model, dat_loader, loss_fn, cuda):
    model.eval()

    loss = 0
    for i, data in enumerate(dat_loader):
        inputs = Variable(data[0], requires_grad=False)
        targets = Variable(data[1], requires_grad=False)
        if cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs = model(inputs)

        loss = loss + loss_fn(outputs, targets).data[0]

    return loss / len(dat_loader)
