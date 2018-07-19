from torch.autograd import Variable
import torch
import numpy as np
import time
import os


class Trainer():

    def __init__(self, model, loss, optimizer, loader_train, patience=None,
                    loader_train_eval=None, loader_val=None, cuda=None,
                    logger=None, data_noise_gaussian=None, save_dir=None):
        """"""
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.patience = patience
        self.loader_train = loader_train
        self.loader_train_eval = loader_train_eval
        self.loader_val = loader_val
        self.cuda = cuda
        self.logger = logger
        self.data_noise_gaussian = data_noise_gaussian
        self.save_dir = save_dir


    def train_epoch(self):
        """ Train model for one epoch"""
        self.model.train()

        if self.cuda:
            self.model.cuda()
            self.loss.cuda()

        total_loss = 0
        for batch_idx, data in enumerate(self.loader_train):

            # add gaussian noise if enabled
            if self.data_noise_gaussian:
                X = data[0].numpy()
                SNR = np.random.uniform(1, 10**2)
                noise = np.random.randn(*X.shape)
                noise_power = np.sum(np.sum(noise ** 2))
                noise = noise / np.sqrt(noise_power)
                X_power = np.sum(np.sum(X ** 2))
                C = X_power / SNR
                X_noise = X + noise * np.sqrt(C)
                data[0] = torch.from_numpy(np.float32( X_noise) )

            inputs = Variable(data[0], requires_grad=False)
            targets = Variable(data[1], requires_grad=False)
            if self.cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
                self.loss = self.loss.cuda()

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, targets)
            loss.backward()
            self.optimizer.step()

            # accumulate loss
            # total_loss += loss.data[0] # before PyTorch 0.4.0
            total_loss += loss.item() # PyTorch 0.4.0 and after

        return total_loss / len(self.loader_train)


    def compute_loss(self, dat_loader):
        """ Compute model loss for provided data loader"""
        self.model.eval()

        if self.cuda:
            self.model.cuda()
            self.loss.cuda()

        total_loss = 0
        for batch_idx, data in enumerate(dat_loader):

            # add gaussian noise
            if self.data_noise_gaussian:
                X = data[0].numpy()
                SNR = np.random.uniform(1, 10**2)
                noise = np.random.randn(*X.shape)
                noise_power = np.sum(np.sum(noise ** 2))
                noise = noise / np.sqrt(noise_power)
                X_power = np.sum(np.sum(X ** 2))
                C = X_power / SNR
                X_noise = X + noise * np.sqrt(C)
                data[0] = torch.from_numpy(np.float32( X_noise) )

            inputs = Variable(data[0], requires_grad=False)
            targets = Variable(data[1], requires_grad=False)
            if self.cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
                self.loss = self.loss.cuda()

            outputs = self.model(inputs)
            loss = self.loss(outputs, targets)

            # accumulate loss
            # total_loss += loss.data[0] # before PyTorch 0.4.0
            total_loss += loss.item() # PyTorch 0.4.0 and after

        return total_loss / len(dat_loader)


    def train(self):
        """Train the model"""
        # initial setup
        epoch = 1
        loss_val_best = 100
        num_epochs_increased = 0
        epoch_best = 1

        # Perform training
        while True:
            # Run one iteration of SGD
            t0 = time.time()
            loss_train = self.train_epoch()
            loss_train_eval = self.compute_loss(self.loader_train_eval)
            loss_val = self.compute_loss(self.loader_val)
            time_epoch = time.time() - t0
            self.logger.add_entry( {'loss_train' : loss_train,
                                'loss_train_eval' : loss_train_eval,
                                'loss_val' : loss_val} )

            # save logger info
            if self.save_dir:
                self.logger.append(os.path.join(self.save_dir, 'log.txt'))

            # change in loss_val
            d_loss_val = (loss_val-loss_val_best)/loss_val_best * 100

            # display results
            print('E: {:} / Train: {:.3e} / Valid: {:.3e} / Diff Valid: {:.2f}% / Diff Valid-Train: {:.1f}% / Time: {:.2f}'.format(epoch, loss_train_eval, loss_val, d_loss_val, (loss_val - loss_train_eval)/loss_train_eval*100, time_epoch))

            # if validation loss improves
            if d_loss_val < -0.05:
                num_epochs_increased = 0

                # record epoch and loss
                epoch_best = epoch
                loss_val_best = loss_val

                # save model weights
                if self.save_dir:
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model.dat'))

            else:
                num_epochs_increased += 1

            # stop training if we lose patience:
            if num_epochs_increased > self.patience:
                break

            # advance epoch counter
            epoch += 1
