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


def run_experiment(exp_name, net: str, seed=None, bs_train=32, bs_test=32, epochs=300, lr=1e-3, l2_reg=.0, num_slices=4):

    if not seed:
        seed = random.randint(0, 2**31)
    torch.manual_seed(seed)
    early_stopping = 30
    cfg = locals()

    train_files = get_files_list('../data/modelnet40_ply_hdf5_2048/train_files.txt')
    test_files = get_files_list('../data/modelnet40_ply_hdf5_2048/test_files.txt')

    loss_fn = F.cross_entropy  # This criterion combines log_softmax and nll_loss in a single function

    if net is 'PointNet':
        our_model = model.PointNet()
        ds_train = training.ModelNet40Ds(train_files)
        ds_test = training.ModelNet40Ds(test_files)
    elif net is 'PicNet':
        our_model = model.PicResNetVox()
        ds_train = training.PicNet40Ds(train_files, num_slices=num_slices)
        ds_test = training.PicNet40Ds(test_files, num_slices=num_slices)
    else:
        loss_fn = F.nll_loss
        our_model = model.CuppNetSumProb()
        ds_train = training.CuppNet40Ds(train_files)
        ds_test = training.CuppNet40Ds(test_files)

    dl_train = DataLoader(ds_train, bs_train, shuffle=True, num_workers=4)
    dl_test = DataLoader(ds_test, bs_test, shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam(our_model.parameters(), lr=lr, weight_decay=l2_reg)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    cfg.update({'optimizer': type(optimizer).__name__})
    cfg.update({'model': str(our_model)})

    for key, val in cfg.items():
        print(f'{key}: {val}')

    if net is 'PicNet':
        trainer = training.NetTrainer(our_model, loss_fn, optimizer, scheduler)
    elif net is 'PointNet':
        trainer = training.PointNetTrainer(our_model, loss_fn, optimizer, scheduler, bn)
    else:
        trainer = training.CuppTrainer(our_model, loss_fn, optimizer, scheduler)

    fit_res = trainer.fit(dl_train, dl_test, epochs, early_stopping=early_stopping, checkpoints=exp_name)
    save_experiment(exp_name, cfg, fit_res)
    return fit_res


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


def send_mail(subject: str, files: list, cfg: dict):
    import smtplib
    from os.path import basename
    from email.mime.application import MIMEApplication
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    recipients = ['moli0389@gmail.com', 'baklawe@yahoo.com']
    from_address = 'cupp3d@gmail.com'

    msg = MIMEMultipart()
    msg['From'] = from_address
    msg['To'] = ', '.join(recipients)
    msg['Subject'] = subject

    text = "\n".join("{}: {}".format(*i) for i in cfg.items())
    msg.attach(MIMEText(text))

    for f in files or []:
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=basename(f)
            )
        # After the file is closed
        part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
        msg.attach(part)

    gmail_pwd = 'stateoftheart'
    smtp = smtplib.SMTP("smtp.gmail.com", 587)
    smtp.ehlo()
    smtp.starttls()
    smtp.login(from_address, gmail_pwd)
    smtp.sendmail(from_address, recipients, msg.as_string())
    smtp.close()
    print('Mail successfully sent')
    return


if __name__ == '__main__':
    net_type = 'PicNet'
    for num_slices in range(2, 16+1):
        expr_name = f'Cuppnet-resnet-{num_slices}-slices'
        fit_results = run_experiment(f'{expr_name}', net_type, lr=5e-4, num_slices=num_slices)
        best_acc = max(fit_results.test_acc)
        exp_cfg, exp_fit_res = load_experiment(f'results/{expr_name}.json')
        send_mail(subject=f'{expr_name} Best: {best_acc:.1f}', files=[f'results/{expr_name}.png'], cfg=exp_cfg)
    # net_type = 'cuppNet'
    # expr_name = f'Cuppnet-resnet'
    # fit_results = run_experiment(f'{expr_name}', net_type, lr=5e-4, num_slices=1)
    # best_acc = max(fit_results.test_acc)
    # exp_cfg, exp_fit_res = load_experiment(f'results/{expr_name}.json')
    # send_mail(subject=f'{expr_name} Best: {best_acc:.1f}', files=[f'results/{expr_name}.png'], cfg=exp_cfg)
