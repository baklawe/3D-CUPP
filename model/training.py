import os
import sys
import tqdm
import torch

from torch.utils.data import DataLoader
from typing import Callable, Any
from typing import NamedTuple, List

import numpy as np
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import platform


class ModelNet40(Dataset):
    def __init__(self, num_samples, augmentations, num_lbo_in, num_lbo_proj, num_nbs, input_phi: bool, overfit: bool):
        super().__init__()
        assert torch.cuda.is_available()
        self.device = torch.device('cuda')
        path = '../Data/faust_synthetic'
        if platform.system() is 'Windows':
            path = '../../Data/faust_synthetic'

        self.dir_dist = path + '/distance_matrix'

    def __getitem__(self, index):
        if self.overfit:
            np.random.seed(index)  # Take the same examples each epoch.
        # Pick shape, isometries and augmentations
        shape_ind = np.random.randint(10)
        # x_ind, y_ind = np.random.randint(10, size=2)
        x_ind, y_ind = np.random.choice(10, 2)
        x0_ind, x1_ind = np.random.choice(self.augmentations, 2)
        y0_ind, y1_ind = np.random.choice(self.augmentations, 2)

        dist_x = self.get_dist_mat(shape_ind, x_ind)
        dist_y = self.get_dist_mat(shape_ind, y_ind)
        nx, ny = dist_x.shape[0], dist_y.shape[0]
        assert dist_x.shape == (nx, nx), f'dist_x.shape={dist_x.shape}'
        assert dist_y.shape == (ny, ny), f'dist_y.shape={dist_y.shape}'

        return x0, x1

    def __len__(self):
        return self.num_samples

    def get_dist_mat(self, shape_ind: int, iso_ind: int) -> torch.Tensor:
        file_name = f'tr_reg_0{shape_ind}{iso_ind}.mat'
        path = os.path.join(self.dir_dist, file_name)
        mat_file = sio.loadmat(path)
        np_file = mat_file['D']
        dist_mat = torch.from_numpy(np_file).float().to(self.device)
        return dist_mat


class BatchResult(NamedTuple):
    corr_loss: float
    fm_loss: float
    dsr_loss: float


class EpochResult(NamedTuple):
    corr_loss: List[float]
    fm_loss: List[float]
    dsr_loss: List[float]


class FitResult(NamedTuple):
    train_corr: List[float]
    train_fm: List[float]
    train_dsr: List[float]
    test_corr: List[float]
    test_fm: List[float]
    test_dsr: List[float]

    def get_dict(self):
        return self._asdict()

    def add_epoch_train(self, epoch_res: EpochResult):
        self.train_corr.append(float(np.mean(epoch_res.corr_loss)))
        self.train_fm.append(float(np.mean(epoch_res.fm_loss)))
        self.train_dsr.append(float(np.mean(epoch_res.dsr_loss)))

    def add_epoch_test(self, epoch_res: EpochResult):
        self.test_corr.append(float(np.mean(epoch_res.corr_loss)))
        self.test_fm.append(float(np.mean(epoch_res.fm_loss)))
        self.test_dsr.append(float(np.mean(epoch_res.dsr_loss)))

    def check_early_stopping(self, early_stopping: int):
        test_lst = self.test_corr[-(early_stopping + 1):]
        return all(earlier <= later for earlier, later in zip(test_lst, test_lst[1:]))


def loss_to_percentage(loss: float) -> float:
    res = (np.sqrt(loss) / 2) * 100
    return res


class IntrinsicLearnerTrainer:
    def __init__(self, model: IntrinsicLearner, loss: IntrinsicLearnerLoss, optimizer):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        assert torch.cuda.is_available()
        self.device = torch.device('cuda')
        model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_test: DataLoader, num_epochs,
            early_stopping, checkpoints: str = None) -> FitResult:

        fit_res = FitResult(train_corr=[], train_fm=[], train_dsr=[], test_corr=[], test_fm=[], test_dsr=[])

        start_epoch = 1
        if checkpoints:
            checkpoint_filename = f'results/{checkpoints}.pt'
            if os.path.isfile(checkpoint_filename):
                print(f'*** Loading checkpoint file {checkpoint_filename}')
                saved_state = torch.load(checkpoint_filename, map_location=self.device)
                self.model.load_state_dict(saved_state['model_state'])
                fit_res = saved_state.get('fit_res', fit_res)
                start_epoch += len(fit_res.test_corr)

        for epoch in range(start_epoch, num_epochs+1):
            print(f'--- EPOCH {epoch}/{num_epochs} ---')

            fit_res.add_epoch_train(self.train_epoch(dl_train))
            fit_res.add_epoch_test(self.test_epoch(dl_test))
            self.plot_error(fit_res)

            if early_stopping and epoch > early_stopping and fit_res.check_early_stopping(early_stopping):
                break

            if checkpoints:
                checkpoint_filename = f'results/{checkpoints}.pt'
                saved_state = dict(model_state=self.model.state_dict(), fit_res=fit_res)
                torch.save(saved_state, checkpoint_filename)
                print(f'*** Saved checkpoint {checkpoints} at epoch {epoch}')

        return fit_res

    def train_epoch(self, dl_train: DataLoader) -> EpochResult:
        self.model.train(True)
        return self._foreach_batch(dl_train, self.train_batch)

    def test_epoch(self, dl_test: DataLoader) -> EpochResult:
        self.model.train(False)
        return self._foreach_batch(dl_test, self.test_batch)

    def train_batch(self, batch) -> BatchResult:
        x0, x1, y0, y1, phi_x, phi_y, dist_x, dist_y, area_x, area_y = batch
        self.optimizer.zero_grad()
        dsr_x0, dsr_x1, dsr_y0, dsr_y1 = self.model.forward(x0, x1, y0, y1, area_x, area_y)
        corr_loss, fm_loss, dsr_loss = self.loss.get_corr_fm_dsr_loss(dsr_x0, dsr_x1, dsr_y0, dsr_y1, dist_x, dist_y, phi_x, phi_y, area_x, area_y)
        loss = corr_loss + self.loss.mu_fm * fm_loss + self.loss.mu_dsr * dsr_loss
        loss.backward()
        self.optimizer.step()
        return BatchResult(corr_loss.item(), fm_loss.item(), dsr_loss.item())

    def test_batch(self, batch) -> BatchResult:
        x0, x1, y0, y1, phi_x, phi_y, dist_x, dist_y, area_x, area_y = batch
        with torch.no_grad():
            dsr_x0, dsr_x1, dsr_y0, dsr_y1 = self.model.forward(x0, x1, y0, y1, area_x, area_y)
            corr_loss, fm_loss, dsr_loss = self.loss.get_corr_fm_dsr_loss(dsr_x0, dsr_x1, dsr_y0, dsr_y1, dist_x, dist_y, phi_x, phi_y, area_x, area_y)
            return BatchResult(corr_loss.item(), fm_loss.item(), dsr_loss.item())

    @staticmethod
    def _foreach_batch(dl: DataLoader, forward_fn: Callable[[Any], BatchResult]) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        fm_losses = []
        dsr_losses = []
        corr_losses = []
        num_batches = len(dl.batch_sampler)

        pbar_file = sys.stdout
        pbar_name = forward_fn.__name__ + " loss"
        with tqdm.tqdm(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                corr_loss, fm_loss, dsr_loss = forward_fn(data)
                percentage = loss_to_percentage(corr_loss)
                pbar.set_description(f'{pbar_name} (fm {fm_loss:.2e}, dsr {dsr_loss:.2e}, '
                                     f'correspondence {corr_loss:.2e}, {percentage:5.2f}%)')
                pbar.update()
                fm_losses.append(fm_loss)
                dsr_losses.append(dsr_loss)
                corr_losses.append(corr_loss)

            avg_fm_loss = sum(fm_losses) / num_batches
            avg_dsr_loss = sum(dsr_losses) / num_batches
            avg_corr_loss = sum(corr_losses) / num_batches
            avg_percentage = loss_to_percentage(avg_corr_loss)
            pbar.set_description(f'{pbar_name} ' f'(fm {avg_fm_loss:.2e}, dsr {avg_dsr_loss:.2e}, '
                                 f'correspondence {avg_corr_loss:.2e}, {avg_percentage:5.2f}%)')
        return EpochResult(corr_losses, fm_losses, dsr_losses)

    @staticmethod
    def plot_error(fit_res: FitResult):
        epochs = [*range(1, len(fit_res.train_corr)+1)]
        fig = plt.figure(1)

        plt.subplot(311)
        plt.plot(epochs, fit_res.train_dsr, color='b')
        plt.plot(epochs, fit_res.test_dsr, color='r')
        plt.yscale('log')
        plt.ylabel('Descriptors')
        # plt.xlabel('#epochs')
        plt.legend(['Train', 'Test'], loc='upper left')

        plt.subplot(312)
        plt.plot(epochs, fit_res.train_fm, color='b')
        plt.plot(epochs, fit_res.test_fm, color='r')
        plt.yscale('log')
        plt.ylabel('Functional Maps')
        # plt.xlabel('#epochs')
        plt.legend(['Train', 'Test'], loc='upper left')

        plt.subplot(313)
        plt.plot(epochs, fit_res.train_corr, color='b')
        plt.plot(epochs, fit_res.test_corr, color='r')
        plt.yscale('log')
        plt.ylabel('Correspondence')
        plt.xlabel('#epochs')
        plt.legend(['Train', 'Test'], loc='upper left')

        plt.subplots_adjust(left=0.2, top=0.95, bottom=0.1, hspace=0.3, wspace=0.3)
        plt.savefig('results/learning_curve.png')
        plt.close(fig)
        return
