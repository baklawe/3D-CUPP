import os
import random
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import training
import model


def get_files_list(list_filename):
    return ['../' + line.rstrip() for line in open(list_filename)]


def run_experiment(exp_name, net: str, seed=None, bs_train=32, bs_test=32, epochs=200, lr=1e-3, l2_reg=1e-3):

    if not seed:
        seed = random.randint(0, 2**31)
    torch.manual_seed(seed)
    early_stopping = max((10, epochs//20))
    cfg = locals()

    train_files = get_files_list('../data/modelnet40_ply_hdf5_2048/train_files.txt')
    test_files = get_files_list('../data/modelnet40_ply_hdf5_2048/test_files.txt')

    if net is 'PointNet':
        our_model = model.PointNet()
        ds_train = training.ModelNet40Ds(train_files)
        ds_test = training.ModelNet40Ds(test_files)
    elif net is 'PicNet':
        our_model = model.PicNet()
        ds_train = training.PicNet40Ds(train_files)
        ds_test = training.PicNet40Ds(test_files)
    else:
        our_model = model.CuppNet()
        ds_train = training.CuppNet40Ds(train_files)
        ds_test = training.CuppNet40Ds(test_files)

    dl_train = DataLoader(ds_train, bs_train, shuffle=True)
    dl_test = DataLoader(ds_test, bs_test, shuffle=True)

    loss_fn = F.nll_loss
    optimizer = torch.optim.Adam(our_model.parameters(), lr=lr, weight_decay=l2_reg)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    cfg.update({'optimizer': type(optimizer).__name__})
    cfg.update({'model': str(our_model)})

    for key, val in cfg.items():
        print(f'{key}: {val}')

    if net is 'PicNet':
        trainer = training.NetTrainer(our_model, loss_fn, optimizer, scheduler)
    elif net is 'PointNet':
        trainer = training.PointNetTrainer(our_model, loss_fn, optimizer, scheduler)
    else:
        trainer = training.CuppTrainer(our_model, loss_fn, optimizer, scheduler)

    fit_res = trainer.fit(dl_train, dl_test, epochs, early_stopping=early_stopping, checkpoints=exp_name)
    save_experiment(exp_name, cfg, fit_res)
    return


def save_experiment(run_name, config, fit_res):
    output = dict(config=config, results=fit_res.get_dict())
    output_filename = f"{os.path.join('results', run_name)}.json"
    with open(output_filename, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'*** Output file {output_filename} written')
    return


def load_experiment(filename):
    with open(filename, 'r') as f:
        output = json.load(f)
    config = output['config']
    fit_res = training.FitResult(**output['results'])
    return config, fit_res


if __name__ == '__main__':
    expr_name = 'PointNet-full2'
    net_type = 'PointNet'
    # net_type = 'CuppNet'
    # net_type = 'PicNet'
    run_experiment(f'{expr_name}', net_type)
    exp_cfg, exp_fit_res = load_experiment(f'results/{expr_name}.json')
