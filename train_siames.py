from    utils.MiniImagenet import *
import os
import sys
from glob import glob

import numpy as np

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
import shutil
import argparse

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from siames import SiameseNet
from one_cycle_policy import OneCyclePolicy
from utils.train_utils import AverageMeter
from tensorboardX import SummaryWriter

root_dir = "/home/few_shot/few_shot_baseline"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_dir = root_dir + "/checkpoint"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
log_dir = root_dir + "/logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def save_checkpoint(state, is_best):

    if is_best:
        filename = './models/best_model.pt'
    else:
        filename = f'./models/model_ckpt_{state["epoch"]}.pt'

    model_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, model_path)

def load_checkpoint(best):
    print(f"[*] Loading model Num....", end="")

    if best:
        model_path = os.path.join(checkpoint_dir, './models/best_model.pt')
    else:
        model_path = sorted(glob(checkpoint_dir + './models/model_ckpt_*.pt'), key=len)[-1]

    ckpt = torch.load(model_path)

    if best:
        print(
            f"Loaded {os.path.basename(model_path)} checkpoint @ epoch {ckpt['epoch']} with best valid acc of {ckpt['best_valid_acc']:.3f}")
    else:
        print(f"Loaded {os.path.basename(model_path)} checkpoint @ epoch {ckpt['epoch']}")

    return ckpt['epoch'], ckpt['best_epoch'], ckpt['best_valid_acc'], ckpt['model_state'], ckpt['optim_state']



def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    epochs = args.epoch    
    task_name = "{}_{}_{}way_{}shot_{}query_{}epoch".format("network","SiamesNetwork",args.n_way,args.k_spt,args.k_qry,epochs)
    checkpoint_path = os.path.join(checkpoint_dir,task_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # batchsz here means total episode number
    imagnet_dir = os.path.join(root_dir,"data/mini-imagenet")
    print("start to load dataset")
    mini = SiamMiniImageNetTrain(imagnet_dir,batchsz=10000,n_way=args.n_way, k_shot=args.k_spt,imsize=args.imgsz)
    mini_val = SiamMiniImageNetTest(imagnet_dir,split="val",batchsz=100,n_way=args.n_way, k_query=args.k_qry, imsize=args.imgsz)
    train_db = DataLoader(mini, args.batch_size, shuffle=True, num_workers=0, pin_memory=True)    
    valid_db = DataLoader(mini_val, 1, shuffle=True, num_workers=0, pin_memory=True)
    print("sucess to load dataset")

# Model, Optimizer, criterion
    model = SiameseNet()
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()

    model.to(device)
    print("success to load model!!")

    # Load check point
    if args.resume:
        start_epoch, best_epoch, best_valid_acc, model_state, optim_state = load_checkpoint(best=False)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optim_state)
        one_cycle = OneCyclePolicy(optimizer, num_steps= args.epoch - start_epoch,
                                   lr_range=(args.lr, 1e-1), momentum_range=(0.85, 0.95))

    else:
        best_epoch = 0
        start_epoch = 0
        best_valid_acc = 0
        one_cycle = OneCyclePolicy(optimizer, num_steps=args.epoch,
                                       lr_range=(args.lr, 1e-1), momentum_range=(0.85, 0.95))

    # create tensorboard summary and add model structure.
    writer = SummaryWriter(log_dir=os.path.join(log_dir,task_name))

    # im1, im2, _ = next(iter(valid_db))
    # input (Batch, Task num, Channel, H, W)
    writer.add_graph(model,
                         [torch.rand((1, 1, 3, 105, 105)).to(device), torch.rand(1, 1, 3, 105, 105).to(device)])

    counter = 0
    num_train = len(train_db)
    num_valid = len(valid_db)
    print(
            f"[*] Train on {len(train_db.dataset)} sample pairs, validate on {len(valid_db.dataset)} trials")

        # Train & Validation
    main_pbar = tqdm(range(start_epoch, args.epoch), initial=start_epoch, position=0,
                         total=args.epoch, desc="Process")
    print("Start to train")
    for epoch in main_pbar:
        train_losses = AverageMeter()
        valid_losses = AverageMeter()

        # TRAIN
        model.train()
        train_pbar = tqdm(enumerate(train_db),total=num_train, desc="Train", position=1, leave=False)
        for i, (x1, x2, y) in train_pbar:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            out = model(x1, x2)
            # loss = criterion(out, y.unsqueeze(1))
            loss = criterion(out, y)

            # compute gradients and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            train_pbar.set_postfix_str(f"loss: {train_losses.val:0.3f}")
            train_losses.update(loss.item(), x1.shape[0])

            # log loss
            writer.add_scalar("Loss/Train", train_losses.val, epoch * len(train_db) + i)
        one_cycle.step()

        # Validation
        if epoch % 3 == 0:

            model.eval()
            correct_sum = 0
            valid_pbar = tqdm(enumerate(valid_db), total=num_valid, desc="Valid", position=1, leave=False)
            with torch.no_grad():
                for i, (x1, x2, y) in valid_pbar:
                    x1, x2, y = x1.to(device), x2.to(device), y.to(device)

                    # compute log probabilities
                    out = model(x1, x2)
                    loss = criterion(out, y)
                    # loss = criterion(out, y.unsqueeze(1))

                    y_pred = torch.sigmoid(out)
                    y_pred = torch.argmax(y_pred)
                    if y_pred == 0:
                        correct_sum += 1

            valid_losses.update(loss.item(), x1.shape[0])

                    # compute acc and log
            valid_acc = correct_sum / num_valid
            writer.add_scalar("Loss/Valid", valid_losses.val, epoch * len(valid_db) + i)
            valid_pbar.set_postfix_str(f"accuracy: {valid_acc:0.3f}")
            writer.add_scalar("Acc/Valid", valid_acc, epoch)

            # check for improvement
            if valid_acc >= best_valid_acc:
                is_best = True
                best_valid_acc = valid_acc
                best_epoch = epoch
                counter = 0
            else:
                is_best = False
                counter += 1

            # checkpoint the model
        if is_best or epoch % 5 == 0 or epoch == args.epoch:
            save_checkpoint(
                {
                        'epoch': epoch,
                        'model_state': model.state_dict(),
                        'optim_state': optimizer.state_dict(),
                        'best_valid_acc': best_valid_acc,
                        'best_epoch': best_epoch,
                }, checkpoint_path, is_best
            )

        main_pbar.set_postfix_str(f"best acc: {best_valid_acc:.3f} best epoch: {best_epoch} ")

        tqdm.write(
                f"[{epoch}] train loss: {train_losses.avg:.3f} - valid loss: {valid_losses.avg:.3f} - valid acc: {valid_acc:.3f} {'[BEST]' if is_best else ''}")

    # release resources
    writer.close()

def save_checkpoint(state, path, is_best):
    filename = '%s/checkpoint.pt' % (path)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('checkpoint.pt', 'best.pt'))

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--batch_size', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--resume', type=str2bool, help='reuse the best checkpoint', default=False)
    argparser.add_argument('--optimizer', type=str, help='set the optimizer', default="ADAM")

    args = argparser.parse_args()

    main(args)