import os
import random
import json

import torch
import torch.nn.functional as F


from torch.utils.data import DataLoader

import training
import model


def point_net_experiment(exp_name, seed=None, bs_train=32, bs_test=32, epochs=7, lr=1e-3, l2_reg=1e-4):

    if not seed:
        seed = random.randint(0, 2**31)
    torch.manual_seed(seed)
    early_stopping = max((10, epochs//20))
    cfg = locals()

    ds_train = training.ModelNet40Train()
    ds_test = training.ModelNet40Test()

    dl_train = DataLoader(ds_train, bs_train, shuffle=True)
    dl_test = DataLoader(ds_test, bs_test, shuffle=True)

    our_model = model.PointNet()
    loss_fn = F.nll_loss
    optimizer = torch.optim.Adam(our_model.parameters(), lr=lr, weight_decay=l2_reg)
    cfg.update({'optimizer': type(optimizer).__name__})
    cfg.update({'model': str(our_model)})

    for key, val in cfg.items():
        print(f'{key}: {val}')

    trainer = training.PointNetTrainer(our_model, loss_fn, optimizer)

    fit_res = trainer.fit(dl_train, dl_test, epochs, early_stopping=early_stopping, checkpoints=exp_name)
    save_experiment(exp_name, cfg, fit_res)


def pic_net_experiment(exp_name, seed=None, bs_train=32, bs_test=32, epochs=50, lr=1e-3, l2_reg=1e-4):

    if not seed:
        seed = random.randint(0, 2**31)
    torch.manual_seed(seed)
    early_stopping = max((10, epochs//20))
    cfg = locals()

    ds_train = training.PicNet40Train()
    ds_test = training.PicNet40Test()

    dl_train = DataLoader(ds_train, bs_train, shuffle=True)
    dl_test = DataLoader(ds_test, bs_test, shuffle=True)

    our_model = model.PicNet()
    loss_fn = F.nll_loss
    optimizer = torch.optim.Adam(our_model.parameters(), lr=lr, weight_decay=l2_reg)
    cfg.update({'optimizer': type(optimizer).__name__})
    cfg.update({'model': str(our_model)})

    for key, val in cfg.items():
        print(f'{key}: {val}')

    trainer = training.PointNetTrainer(our_model, loss_fn, optimizer)

    fit_res = trainer.fit(dl_train, dl_test, epochs, early_stopping=early_stopping, checkpoints=exp_name)
    save_experiment(exp_name, cfg, fit_res)


def save_experiment(run_name, config, fit_res):
    output = dict(
        config=config,
        results=fit_res.get_dict()
    )
    output_filename = f"{os.path.join('results', run_name)}.json"
    with open(output_filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'*** Output file {output_filename} written')


def load_experiment(filename):
    with open(filename, 'r') as f:
        output = json.load(f)

    config = output['config']
    fit_res = training.FitResult(**output['results'])
    return config, fit_res


if __name__ == '__main__':
    expr_name = 'pic_net_try'
    pic_net_experiment(f'{expr_name}')
    exp_cfg, exp_fit_res = load_experiment(f'results/{expr_name}.json')
