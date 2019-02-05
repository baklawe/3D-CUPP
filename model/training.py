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


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return data, label


class ModelNet40Train(Dataset):
    def __init__(self):
        super().__init__()
        self.num_points = 1024
        self.examples = []
        for ind in range(5):
            current_data, current_label = load_h5(f'../data/modelnet40_ply_hdf5_2048/ply_data_train{ind}.h5')
            current_data = current_data[:, 0:self.num_points, :]
            self.examples.append((current_data, current_label))

    def __getitem__(self, index):
        # Get file name
        file_ind = index // 2048
        example_ind = index % 2048
        item, label = self.examples[file_ind]
        item, label = item[example_ind, :, :].transpose(), label[example_ind, :]
        assert item.shape == (3, self.num_points), f'item.shape={item.shape}'
        assert label.shape == (1,), f'label.shape={label.shape}'
        item_tensor = torch.from_numpy(item).float()
        label_tensor = torch.from_numpy(label).long()
        return item_tensor, label_tensor

    def __len__(self):
        """
        modelnet40 train composed of 5 h5 files, each contains 2048 examples except for the last one which has 1648
        """
        return 4 * 2048 + 1648


class ModelNet40Test(Dataset):
    def __init__(self):
        super().__init__()
        self.num_points = 1024
        self.examples = []
        for ind in range(2):
            current_data, current_label = load_h5(f'../data/modelnet40_ply_hdf5_2048/ply_data_test{ind}.h5')
            current_data = current_data[:, 0:self.num_points, :]
            self.examples.append((current_data, current_label))

    def __getitem__(self, index):
        # Get file name
        file_ind = index // 2048
        example_ind = index % 2048
        item, label = self.examples[file_ind]
        item, label = item[example_ind, :, :].transpose(), label[example_ind, :]
        assert item.shape == (3, self.num_points), f'item.shape={item.shape}'
        assert label.shape == (1,), f'label.shape={label.shape}'
        item_tensor = torch.from_numpy(item).float()
        label_tensor = torch.from_numpy(label).long()
        return item_tensor, label_tensor

    def __len__(self):
        """
        modelnet40 train composed of 5 h5 files, each contains 2048 examples except for the last one which has 1648
        """
        return 2048 + 420


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
        test_lst = self.test_acc[-(early_stopping + 1):]
        return all(earlier <= later for earlier, later in zip(test_lst, test_lst[1:]))


class PointNetTrainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        assert torch.cuda.is_available()
        self.device = torch.device('cuda')
        model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_test: DataLoader, num_epochs,
            early_stopping, checkpoints: str = None) -> FitResult:

        fit_res = FitResult(train_loss=[], train_acc=[], test_loss=[], test_acc=[])

        start_epoch = 1
        if checkpoints:
            checkpoint_filename = f'results/{checkpoints}.pt'
            if os.path.isfile(checkpoint_filename):
                print(f'*** Loading checkpoint file {checkpoint_filename}')
                saved_state = torch.load(checkpoint_filename, map_location=self.device)
                self.model.load_state_dict(saved_state['model_state'])
                fit_res = saved_state.get('fit_res', fit_res)
                start_epoch += len(fit_res.test_loss)

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

                pbar.set_description(f'{pbar_name} (Loss: {batch_res.loss:.3e},'
                                     f' Correct: {batch_res.num_correct})')
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            assert (num_correct / num_samples) < 1, f'num_correct={num_correct}, num_samples={num_samples}'
            accuracy = 100. * (num_correct / num_samples)
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3e}, '
                                 f'Accuracy {accuracy:.1f})')

        return EpochResult(losses=losses, accuracy=accuracy)

    @staticmethod
    def plot_error(fit_res: FitResult):
        epochs = [*range(1, len(fit_res.train_acc)+1)]
        # batches = [*range(1, len(fit_res.train_loss)+1)]
        fig = plt.figure(1)

        plt.subplot(211)
        plt.plot(epochs, fit_res.train_loss, color='b')
        plt.plot(epochs, fit_res.test_loss, color='r')
        plt.yscale('log')
        plt.ylabel('Loss')
        # plt.xlabel('#batches')
        plt.legend(['Train', 'Test'], loc='upper left')

        plt.subplot(212)
        plt.plot(epochs, fit_res.train_acc, color='b')
        plt.plot(epochs, fit_res.test_acc, color='r')
        plt.yscale('log')
        plt.ylabel('Accuracy')
        plt.xlabel('#epochs')
        plt.legend(['Train', 'Test'], loc='upper left')

        plt.subplots_adjust(left=0.2, top=0.95, bottom=0.1, hspace=0.3, wspace=0.3)
        plt.savefig('results/learning_curve.png')
        plt.close(fig)
        return
