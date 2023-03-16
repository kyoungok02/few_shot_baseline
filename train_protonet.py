from    utils.MiniImagenet import *
import os
import sys
from glob import glob

import numpy as np

import torch
import torch.optim as optim
# from tqdm import tqdm
from torch.utils.data import DataLoader
import shutil
import argparse

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.protonet import *
from utils.prototypical_loss import PrototypicalLoss
from utils.utils import *
from utils.train_utils import AverageMeter
from utils.resnet import *
from utils.Cifar100FS import *
from tensorboardX import SummaryWriter

# GPU setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_checkpoint(state, path, is_best):
    filename = '%s/checkpoint.pt' % (path)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('checkpoint.pt', 'best.pt'))

def load_checkpoint(best, checkpoint_dir):
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
    
    set_seed(222)
    
    # make dir(필요한 directory 생성)
    epochs = args.epoch
    root_dir = args.root_dir
    # check point directory
    checkpoint_dir = root_dir + "/checkpoint"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # log directory
    log_dir = root_dir + "/logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # project folder name
    task_name = "{}_{}_{}_{}way_{}shot_{}query_{}epoch".format(args.model, args.dataset,"ProtoNet",args.n_way,args.k_spt,args.k_qry,epochs)
    checkpoint_path = os.path.join(checkpoint_dir,task_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    
    # batchsz here means total episode number
    print("start to load dataset")
    if args.dataset =="mini-imagenet":
        imagnet_dir = os.path.join(root_dir,"data/mini-imagenet")
        print(f"Dataset setting : mini-imagenet \n The dataset dir : {imagnet_dir}")
        train_dataset = MetaMiniImageNet(imagnet_dir,split="train",batchsz=10000,n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, imsize=args.imgsz)
        valid_dataset= MetaMiniImageNet(imagnet_dir,split="val",batchsz=100,n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, imsize=args.imgsz)
        train_db = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        valid_db = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    elif args.dataset == "cifar100":
        cifar_dir = os.path.join(root_dir,"data/fs_fc100")
        print(f"Dataset setting : few-shot cifar 100 \n The dataset dir : {cifar_dir}")
        train_dataset = MetaCifer100FS(cifar_dir,split="train",batchsz=10000,n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, imsize=args.imgsz)
        valid_dataset= MetaCifer100FS(cifar_dir,split="val",batchsz=100,n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, imsize=args.imgsz)
        train_db = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        valid_db = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print("sucess to load dataset")

# Model, Optimizer, criterion
    print("start to load model")
    if args.model == 'Baseline':
        model = ProtoNet(input_dim=3)
    elif args.model == 'Resnet34':
        model = ProtoNet_withResNet(num_classes=1024)
    optimizer = optim.Adam(model.parameters())
    criterion = PrototypicalLoss().to(device)
    model.to(device)
    print("success to load model!!")

    # Load check point
    if args.resume:
        start_epoch, best_epoch, best_valid_acc, model_state, optim_state = load_checkpoint(best=False, checkpoint_dir=checkpoint_dir)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optim_state)
    else:
        best_epoch = 0
        best_valid_acc = 0

    # create tensorboard summary and add model structure.
    writer = SummaryWriter(log_dir=os.path.join(log_dir,task_name))

    # im1, im2, _ = next(iter(valid_db))
    # # input (Batch, Task num, Channel, H, W)
    # writer.add_graph(model,
    #                      [torch.rand((1, 1, 3, 105, 105)).to(device), torch.rand(1, 1, 3, 105, 105).to(device)])

    # print(
    #         f"[*] Train on {len(train_db.dataset)} sample pairs, validate on {len(valid_db.dataset)} trials")
    print(
            f"[*] Train on {len(train_dataset)} sample pairs, validate on {len(valid_dataset)} trials")

        # Train & Validation
    print("Start to train")
    for epoch in range(args.epoch):
        train_losses = AverageMeter()
        valid_losses = AverageMeter()
        valid_acc = AverageMeter()

        # TRAIN
        model.train()
        # tqdm 사용시
        # train_pbar = tqdm(enumerate(train_db),total=num_train, desc="Train", position=1, leave=False)
        correct_sum = 0

        for i, (sup_x, sup_y, query_x, query_y) in enumerate(train_db):
        # tqdm 사용시
        # for i, (x1, x2, y) in train_pbar:
        # batch 사용 안할 때
        # for i, (sup_x, sup_y, query_x, query_y) in enumerate(train_dataset):
            # sup_x, sup_y, query_x, query_y = sup_x.unsqueeze(0),sup_y.unsqueeze(0),query_x.unsqueeze(0),query_y.unsqueeze(0)
            sup_x, sup_y, query_x, query_y = sup_x.to(device),sup_y.to(device),query_x.to(device),query_y.to(device)
            sup_out = model(sup_x)
            q_out = model(query_x)
            loss, result = criterion(sup_out, sup_y, q_out, query_y,device)
            loss.backward()
            optimizer.step()
            
            # print the result
            train_losses.update(result['loss'])
            print(f"step: {i+1}, loss: {train_losses.val:0.3f}, acc : {result['acc']}")
            
            # compute gradients and update
            optimizer.zero_grad()

            # log loss
            writer.add_scalar("Loss/Train", train_losses.val, epoch * len(train_db) + i)

        # Validation
        if epoch % 3 == 0:

            model.eval()
            # tqdm 사용시
            # valid_pbar = tqdm(enumerate(valid_db), total=num_valid, desc="Valid", position=1, leave=False)
            with torch.no_grad():
                for i, (sup_x, sup_y, query_x, query_y) in enumerate(valid_db):
                # tqdm 사용시
                # for i, (sup_x, sup_y, query_x, query_y) in valid_pbar:
                # batch 사용 안할 때
                # for i, (sup_x, sup_y, query_x, query_y) in enumerate(train_dataset):
                #     sup_x, sup_y, query_x, query_y = sup_x.unsqueeze(0),sup_y.unsqueeze(0),query_x.unsqueeze(0),query_y.unsqueeze(0)
                    sup_x, sup_y, query_x, query_y = sup_x.to(device),sup_y.to(device),query_x.to(device),query_y.to(device)

                    # compute log probabilities
                    sup_out = model(sup_x)
                    q_out = model(query_x)
                    loss, result = criterion(sup_out, sup_y, q_out, query_y,device)
                    valid_losses.update(result['loss'])
                    valid_acc.update(result['acc'])
                
                    # compute acc and log
            writer.add_scalar("Loss/Valid", valid_losses.val, epoch * len(valid_db) + i)
            # tqdm으로 acc 계산
            # valid_pbar.set_postfix_str(f"accuracy: {valid_acc:0.3f}")
            # batch 사용 안할 때
            # writer.add_scalar("Loss/Valid", valid_losses.val, epoch * len(valid_dataset) + i)
            print(f"accuracy: {valid_acc.avg:0.3f}")
            writer.add_scalar("Acc/Valid", valid_acc.avg, epoch)

            # check for improvement
            if valid_acc.val >= best_valid_acc:
                is_best = True
                best_valid_acc = valid_acc.val
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

        print(f"best acc: {best_valid_acc:.3f} best epoch: {best_epoch} ")
        # tqdm 사용 시
        # tqdm.write(
        #         f"[{epoch}] train loss: {train_losses.avg:.3f} - valid loss: {valid_losses.avg:.3f} - valid acc: {valid_acc.avg:.3f} {'[BEST]' if is_best else ''}")
        print(
                f"[{epoch}] train loss: {train_losses.avg:.3f} - valid loss: {valid_losses.avg:.3f} - valid acc: {valid_acc.avg:.3f} {'[BEST]' if is_best else ''}")

    # release resources
    writer.close()

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
    # model argument
    argparser.add_argument('--seed', type=int, help='initial seed number', default=7)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--batch_size', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--resume', type=str2bool, help='reuse the best checkpoint', default=False)
    # configuratioin
    argparser.add_argument('--root_dir', type=str, help='main project directory path', default="/Users/kyoung-okyang/few_shot/few_shot_baseline")
    argparser.add_argument('--dataset', type=str, help='set the dataset', default='mini-imagenet')
    argparser.add_argument('--model', type=str, help='baseline model', default='Baseline')

    args = argparser.parse_args()

    main(args)
