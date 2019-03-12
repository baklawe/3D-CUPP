import os
import sys
import tqdm
import torch
import h5py

from torch.utils.data import DataLoader
from typing import Callable, Any
from typing import NamedTuple, List

from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import torch.nn.functional as F


class ModelNet40Ds(Dataset):
    def __init__(self, h5_files: List[str]):
        super().__init__()
        self.num_points = 1024
        self.tot_examples = 0
        self.examples_per_file = 2048
        self.examples = []
        for h5_file in h5_files:
            current_data, current_label = self.load_h5(h5_file)
            current_data = current_data[:, 0:self.num_points, :]
            self.examples.append((current_data, current_label))
            self.tot_examples += current_data.shape[0]
        self.train = 'train' in h5_files[0]

    def __getitem__(self, index):
        item, label = self.get_np_pc(index)
        if self.train:
            item = self.jitter_pc(self.rotate_pc(item))
        item_tensor = torch.from_numpy(item).float()
        label_tensor = torch.from_numpy(label).long()
        return item_tensor, label_tensor

    def __len__(self):
        return self.tot_examples

    def get_np_pc(self, index):
        # Get file name
        file_ind = index // self.examples_per_file
        example_ind = index % self.examples_per_file
        item, label = self.examples[file_ind]
        item, label = item[example_ind, :, :].transpose(), label[example_ind, :]
        assert item.shape == (3, self.num_points), f'item.shape={item.shape}'
        assert label.shape == (1,), f'label.shape={label.shape}'
        return item, label

    @staticmethod
    def load_h5(h5_filename):
        f = h5py.File(h5_filename)
        data = f['data'][:]
        label = f['label'][:]
        return data, label

    @staticmethod
    def rotate_pc(pc):
        """
        :param pc: (3, N)
        :return: pc rotated around y axis.
        """
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_y = np.array([[cosval, 0, sinval],
                               [0, 1, 0],
                               [-sinval, 0, cosval]])
        return rotation_y @ pc

    @staticmethod
    def jitter_pc(pc, sigma=0.01, clip=0.05):
        assert (clip > 0)
        noise = np.clip(sigma * np.random.randn(*pc.shape), -1 * clip, clip)
        return pc + noise


class PicNet40Ds(ModelNet40Ds):
    def __init__(self, h5_files: List[str], num_slices=3):
        super().__init__(h5_files)
        self.size = 128
        self.r_lst = [(0, 0, 0), (-np.pi, 0, 0),
                      (0, 0.5*np.pi, 0), (0, -0.5*np.pi, 0), (0.5*np.pi, 0, 0), (-0.5*np.pi, 0, 0)]
        self.m = len(self.r_lst)
        self.t_vec = (0, 0, 5)
        self.num_slices = num_slices

    def normalize_im(self, im_points):
        # if self.train:
        # im_points = self.rot_2d(im_points)
        # print(f'np.min(im_points)={np.min(im_points)}')
        im_points += np.array([0.25, 0.25])
        im_points *= np.array([float(self.size - 1) / 0.5])
        return np.int32(im_points)

    def rot_2d(self, pts):
        # rotation_angle = np.random.choice([i * 0.5 * np.pi for i in range(4)])
        rotation_angle = np.random.uniform(low=0, high=2*np.pi)

        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rot = np.array([[cosval, -sinval],
                        [sinval, cosval]])
        return pts @ rot

    def pc_to_im(self, pc, rvec):
        camera_dist = np.random.uniform(low=4.5, high=5.5)
        t_vec = (0, 0, camera_dist)
        # if self.train:
        #     t_vec[2] += np.random.uniform(low=-0.5, high=0.5)
        image_points, _ = cv.projectPoints(pc, rvec=rvec,
                                           tvec=t_vec,
                                           cameraMatrix=np.eye(3), distCoeffs=np.zeros((4,)))

        image_points = self.normalize_im(image_points[:, 0, :])
        im = np.zeros((self.size, self.size))
        im[image_points[:, 0], image_points[:, 1]] = 1
        return im

    def pc_to_mass_im(self, pc, rvec):
        image_points, _ = cv.projectPoints(pc, rvec=rvec,
                                           tvec=self.t_vec,
                                           cameraMatrix=np.eye(3), distCoeffs=np.zeros((4,)))
        image_points = self.normalize_im(image_points[:, 0, :])
        im = np.zeros((self.size * self.size))
        idx = image_points[:, 0] * self.size + image_points[:, 1]
        im += np.bincount(idx, minlength=self.size * self.size)
        im = im.reshape(self.size, self.size) / np.sum(im)
        return im

    @staticmethod
    def get_rot_mat(axis, theta):
        cosval = np.cos(theta)
        sinval = np.sin(theta)
        if axis == 0:
            return np.array([[1, 0, 0],
                             [0, cosval, -sinval],
                             [0, sinval, cosval]])
        elif axis == 1:
            return np.array([[cosval, 0, sinval],
                             [0, 1, 0],
                             [-sinval, 0, cosval]])
        else:
            return np.array([[cosval, -sinval, 0],
                             [sinval, cosval, 0],
                             [0, 0, 1]])

    def get_im_slices(self, pc, num_slices: int):
        """
        :param pc: (N, 3)
        :param num_slices:
        :return: (num_slices, S, S)
        """
        z = pc[:, 2]
        delta = (np.max(z) - np.min(z)) / num_slices
        bounds = [np.min(z) + i * delta for i in range(num_slices + 1)]
        im = np.empty((num_slices, self.size, self.size))
        for i, (lower, upper) in enumerate(zip(bounds, bounds[1:])):
            idx = np.logical_and(z >= lower, z <= upper)
            # idx = z >= lower
            if np.sum(idx) == 0:
                im[i, :, :] = np.zeros((self.size, self.size))
            else:
                im[i, :, :] = self.pc_to_im(pc[idx, :], rvec=(0, 0, 0))
        return im

    def get_im_rgb(self, pc):
        """
        :param pc: (N, 3)
        :return: list contains num_slices images
        """
        z = pc[:, 2]
        delta = (np.max(z) - np.min(z)) / 3
        bounds = [np.min(z) + i * delta for i in range(3+1)]
        im = np.empty((3, self.size, self.size))
        for i, (lower, upper) in enumerate(zip(bounds, bounds[1:])):
            idx = np.logical_and(z >= lower, z <= upper)
            if np.sum(idx) == 0:
                im[i, :, :] = np.zeros((self.size, self.size))
            else:
                im[i, :, :] = self.pc_to_im(pc[idx, :], rvec=(0, 0, 0))
        return im

    def __getitem__(self, index):
        pc, label = self.get_np_pc(index)
        if self.train:
            pc = self.jitter_pc(pc)
            # pc = self.jitter_pc(self.rotate_pc(pc))
        # im_list = [self.pc_to_im(pc.transpose(), rvec=rv) for rv in self.r_lst]
        # rot_lst = [(0, 0), (1, 0.5 * np.pi), (0, 0.5 * np.pi)]
        rot_lst = [(0, 0), (1, 0.5 * np.pi), (0, 0.5 * np.pi), (0, -np.pi), (1, -0.5 * np.pi), (0, -0.5 * np.pi)]
        im_list = []
        for axis, theta in rot_lst:
            rot = self.get_rot_mat(axis=axis, theta=theta)
            pc_rot = (rot @ pc).transpose()
            im_list.append(self.get_im_slices(pc_rot, num_slices=self.num_slices))
        # im_list = [self.pc_to_mass_im(pc.transpose(), rvec=rv) for rv in self.r_lst]
        # Create (size, size, M) numpy array
        # im_item = np.stack(im_list, axis=-1)[np.newaxis, ...]
        im_item = np.stack(im_list, axis=-1)
        # assert im_item.shape == (1, self.size, self.size, self.m), f'im_item.shape={im_item.shape}'
        assert im_item.shape == (self.num_slices, self.size, self.size, len(rot_lst)), f'im_item.shape={im_item.shape}'
        im_tensor = torch.from_numpy(im_item).float()
        label_tensor = torch.from_numpy(label).long()
        return im_tensor, label_tensor


class CuppNet40Ds(PicNet40Ds):
    def __init__(self, h5_files: List[str]):
        super().__init__(h5_files)

    def __getitem__(self, index):
        pc, label = self.get_np_pc(index)

        if self.train:
            pc = self.jitter_pc(pc)
        rot_lst = [(0, 0), (1, 0.5 * np.pi), (0, 0.5 * np.pi), (0, -np.pi), (1, -0.5 * np.pi), (0, -0.5 * np.pi)]
        im_list = []
        for axis, theta in rot_lst:
            rot = self.get_rot_mat(axis=axis, theta=theta)
            pc_rot = (rot @ pc).transpose()
            im_list.append(self.get_im_slices(pc_rot, num_slices=1))
        im_item = np.stack(im_list, axis=-1)

        # im_list = [self.pc_to_im(pc.transpose(), rvec=rv) for rv in self.r_lst]
        # Create (size, size, M) numpy array
        # im_item = np.stack(im_list, axis=-1)[np.newaxis, ...]

        assert im_item.shape == (1, self.size, self.size, self.m), f'im_item.shape={im_item.shape}'
        im_tensor = torch.from_numpy(im_item).float()
        if self.train:
            # pc = self.jitter_pc(self.rotate_pc(pc))
            pc = self.rotate_pc(pc)
        pc_tensor = torch.from_numpy(pc).float()
        label_tensor = torch.from_numpy(label).long()
        return pc_tensor, im_tensor, label_tensor


class BatchResult(NamedTuple):
    loss: float
    num_correct: int


class EpochResult(NamedTuple):
    losses: List[float]
    accuracy: float


class FitResult(NamedTuple):
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]

    def get_dict(self):
        return self._asdict()

    def add_epoch_train(self, epoch_res: EpochResult):
        self.train_loss.append(sum(epoch_res.losses)/len(epoch_res.losses))
        self.train_acc.append(epoch_res.accuracy)

    def add_epoch_test(self, epoch_res: EpochResult):
        self.test_loss.append(sum(epoch_res.losses)/len(epoch_res.losses))
        self.test_acc.append(epoch_res.accuracy)

    def check_early_stopping(self, early_stopping: int):
        ewb = len(self.test_acc) - (np.argmax(self.test_acc) + 1)
        return ewb > early_stopping


class NetTrainer:
    def __init__(self, model, loss_fn, optimizer, scheduler, bn_decay: bool = False, min_lr: float = 1e-4):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        assert torch.cuda.is_available()
        self.device = torch.device('cuda')
        model.to(self.device)
        self.exp_name = None
        self.bn_decay = bn_decay
        self.min_lr = min_lr
        self.confusion_mat = np.zeros([40 * 40])
        self.confusion_mat_list = []

    def update_bn_momentum(self, epoch):
        # Update BatchNorm momentum, start with 0.5 and every 20 epochs multiply by 0.5, clip to 0.01
        momentum = max([0.01, 0.5 * 0.5 ** (epoch // 20)])
        for seq in self.model.children():
            for layer in seq.children():
                if type(layer) is torch.nn.BatchNorm1d:
                    layer.momentum = momentum
        return momentum

    def optimizer_lr_step(self):
        lr = self.scheduler.get_lr()[0]
        if lr > self.min_lr:
            self.scheduler.step()
            lr = self.scheduler.get_lr()[0]
        return lr

    def fit(self, dl_train: DataLoader, dl_test: DataLoader, num_epochs,
            early_stopping, checkpoints: str = None) -> FitResult:

        fit_res = FitResult(train_loss=[], train_acc=[], test_loss=[], test_acc=[])

        start_epoch = 1
        if checkpoints:
            self.exp_name = checkpoints
            # checkpoint_filename = f'results/{checkpoints}.pt'
            # if os.path.isfile(checkpoint_filename):
            #     print(f'*** Loading checkpoint file {checkpoint_filename}')
            #     saved_state = torch.load(checkpoint_filename, map_location=self.device)
            #     self.model.load_state_dict(saved_state['model_state'])
            #     fit_res = saved_state.get('fit_res', fit_res)
            #     start_epoch += len(fit_res.test_loss)
            #     for i in range(start_epoch):
            #         self.optimizer_lr_step()

        for epoch in range(start_epoch, num_epochs+1):
            lr = self.optimizer_lr_step()

            if self.bn_decay:
                momentum = self.update_bn_momentum(epoch=epoch-1)
                print(f'--- EPOCH {epoch}/{num_epochs}, LR: {lr:.3e}, BN: {momentum:.3f}')
            else:
                print(f'--- EPOCH {epoch}/{num_epochs}, LR: {lr:.3e}')

            fit_res.add_epoch_train(self.train_epoch(dl_train))
            self.confusion_mat = np.zeros([40 * 40])
            fit_res.add_epoch_test(self.test_epoch(dl_test))
            self.plot_error(fit_res)
            mat = self.confusion_mat.reshape(40, 40)
            self.confusion_mat_list.append(mat)
            if len(self.confusion_mat_list) > 10:
                del self.confusion_mat_list[0]
            self.plot_confusion_mat()

            if early_stopping and epoch > early_stopping and fit_res.check_early_stopping(early_stopping):
                break

            # if checkpoints:
            #     checkpoint_filename = f'results/{checkpoints}.pt'
            #     saved_state = dict(model_state=self.model.state_dict(), fit_res=fit_res)
            #     torch.save(saved_state, checkpoint_filename)
            #     print(f'*** Saved checkpoint {checkpoints} at epoch {epoch}')
        self.plot_error(fit_res)
        return fit_res

    def train_epoch(self, dl_train: DataLoader) -> EpochResult:
        self.model.train(True)
        return self._foreach_batch(dl_train, self.train_batch)

    def test_epoch(self, dl_test: DataLoader) -> EpochResult:
        self.model.train(False)
        return self._foreach_batch(dl_test, self.test_batch)

    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x, y = x.to(self.device), y.view(-1,).to(self.device)
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()
        num_correct = torch.sum(y == torch.argmax(y_pred, dim=-1).view(-1,))
        return BatchResult(loss.item(), num_correct.item())

    def test_batch(self, batch) -> BatchResult:
        with torch.no_grad():
            x, y = batch
            x, y = x.to(self.device), y.view(-1,).to(self.device)
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)
            num_correct = torch.sum(y == torch.argmax(y_pred, dim=-1).view(-1,))

            pred = torch.argmax(y_pred, dim=-1).view(-1, )
            y_idx = y != pred
            idx = y[y_idx] * 40 + pred[y_idx]
            self.confusion_mat += np.bincount(idx.cpu().numpy(), minlength=40 * 40)

            return BatchResult(loss.item(), num_correct.item())

    @staticmethod
    def _foreach_batch(dl: DataLoader, forward_fn: Callable[[Any], BatchResult]) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_batches = len(dl.batch_sampler)
        num_samples = len(dl.sampler)

        pbar_file = sys.stdout
        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)
                pbar.set_description(f'{pbar_name} (Loss: {batch_res.loss:.3e}, Correct: {batch_res.num_correct})')
                pbar.update()
                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct
            avg_loss = sum(losses) / num_batches
            accuracy = 100. * (num_correct / num_samples)
            pbar.set_description(f'{pbar_name} (Avg. Loss {avg_loss:.3e}, Accuracy {accuracy:.1f})')
        return EpochResult(losses=losses, accuracy=accuracy)

    def plot_error(self, fit_res: FitResult):
        epochs = [*range(1, len(fit_res.train_acc)+1)]
        plt.rcParams.update({'font.size': 18})
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 10), sharex='col')
        best_acc_train = max(fit_res.train_acc)
        best_acc_test = max(fit_res.test_acc)

        ax = axes[0]
        ewb = len(fit_res.test_acc) - (np.argmax(fit_res.test_acc) + 1)
        ax.set_title(f'{self.exp_name}, Best Train: {best_acc_train:.1f}, Best Test: {best_acc_test:.1f}, ewb: {ewb}')
        ax.plot(epochs, fit_res.train_loss, color='b')
        ax.plot(epochs, fit_res.test_loss, color='r')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.legend(['Train', 'Test'], loc='upper left')
        ax.grid(axis='y')

        ax = axes[1]
        ax.plot(epochs, fit_res.train_acc, color='b')
        ax.plot(epochs, fit_res.test_acc, color='r')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('#epochs')
        ax.legend(['Train', 'Test'], loc='upper left')
        ax.grid(axis='y')
        ax.set_yticks(range(50, 100, 5))
        plt.savefig('results/learning_curve.png')
        plt.savefig(f'results/{self.exp_name}.png')
        plt.close(fig)
        return

    def plot_confusion_mat(self):
        mat = sum(self.confusion_mat_list)
        mat = mat / np.sum(mat)
        mat = np.ma.masked_where(mat < 0.01, mat)
        cmap = plt.cm.OrRd
        cmap.set_bad(color='white')
        fig = plt.figure(figsize=(16, 10))
        plt.imshow(mat, cmap=cmap)
        plt.xticks(np.arange(40), fontsize=6)
        plt.yticks(np.arange(40), fontsize=6)
        plt.grid()
        plt.colorbar()
        plt.title(f'Confusion matrix - {self.exp_name}')
        plt.savefig(f'results/confusion_mat_{self.exp_name}.png')
        plt.close(fig)
        return


class PointNetTrainer(NetTrainer):
    def __init__(self, model, loss_fn, optimizer, scheduler, bn_decay=True):
        super().__init__(model, loss_fn, optimizer, scheduler, bn_decay=bn_decay)
        self.mu = 0.001

    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x, y = x.to(self.device), y.view(-1,).to(self.device)
        self.optimizer.zero_grad()
        y_pred, trans64 = self.model(x)
        eye = torch.eye(64, device=x.device).view(1, 64, 64)
        loss = self.loss_fn(y_pred, y) + self.mu * F.mse_loss(trans64.bmm(trans64.transpose(1, 2)), eye)
        loss.backward()
        self.optimizer.step()
        num_correct = torch.sum(y == torch.argmax(y_pred, dim=-1).view(-1,))
        return BatchResult(loss.item(), num_correct.item())

    def test_batch(self, batch) -> BatchResult:
        with torch.no_grad():
            x, y = batch
            x, y = x.to(self.device), y.view(-1, ).to(self.device)
            y_pred, trans64 = self.model(x)
            eye = torch.eye(64, device=x.device).view(1, 64, 64)
            loss = self.loss_fn(y_pred, y) + self.mu * F.mse_loss(trans64.bmm(trans64.transpose(1, 2)), eye)
            num_correct = torch.sum(y == torch.argmax(y_pred, dim=-1).view(-1,))
            return BatchResult(loss.item(), num_correct.item())


class CuppTrainer(PointNetTrainer):
    def __init__(self, model, loss_fn, optimizer, scheduler):
        super().__init__(model, loss_fn, optimizer, scheduler)

    def train_batch(self, batch) -> BatchResult:
        pc, proj, y = batch
        pc, proj, y = pc.to(self.device), proj.to(self.device), y.view(-1,).to(self.device)
        self.optimizer.zero_grad()
        y_pred, trans64 = self.model(pc, proj)
        eye = torch.eye(64, device=y.device).view(1, 64, 64)
        loss = self.loss_fn(y_pred, y) + self.mu * F.mse_loss(trans64.bmm(trans64.transpose(1, 2)), eye)
        loss.backward()
        self.optimizer.step()
        num_correct = torch.sum(y == torch.argmax(y_pred, dim=-1).view(-1,))
        return BatchResult(loss.item(), num_correct.item())

    def test_batch(self, batch) -> BatchResult:
        with torch.no_grad():
            pc, proj, y = batch
            pc, proj, y = pc.to(self.device), proj.to(self.device), y.view(-1, ).to(self.device)
            y_pred, trans64 = self.model(pc, proj)
            eye = torch.eye(64, device=y.device).view(1, 64, 64)
            loss = self.loss_fn(y_pred, y) + self.mu * F.mse_loss(trans64.bmm(trans64.transpose(1, 2)), eye)
            num_correct = torch.sum(y == torch.argmax(y_pred, dim=-1).view(-1,))
            return BatchResult(loss.item(), num_correct.item())



