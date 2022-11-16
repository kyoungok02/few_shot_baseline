from utils.resnet import *
from utils.MiniImagenet import *
from utils.utils import *
from utils.logger import *
from utils.Models import *

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse

import time
from copy import deepcopy
from collections import OrderedDict


from tensorboardX import SummaryWriter

root_dir = "/home/few_shot/few_shot_baseline"
# device = torch.device('mps' if torch.has_mps else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_dir = root_dir + "/checkpoint"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
log_dir = root_dir + "/logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def inner_loop(args, meta_learner, support_x, support_y, query_x, query_y, logger, iter_counter, mode):
    '''
    run a single episode == n-way k-shot problem
    '''
    tuned_params = OrderedDict({})

    # only fine-tune the conv/fc weights/biases. do not fine-tuned the bn params in original paper
    # for k, v in meta_learner.named_parameters():
    #     if ('conv' in k) or ('fc' in k):
    #         tuned_params[k] = v.clone()

    # in baselines paper they fine-tuned all bn params
    for k, v in meta_learner.named_parameters():
        tuned_params[k] = v.clone()
        
    # decoupling the base/meta learner makes faster 2nd order calc
    # decoupling check!
    # if args.decoupled:
    #     tuned_params= OrderedDict(
    #         [(k,tuned_params[k]) 
    #         for k in tuned_params.keys()
    #         if 'fc' in k]
    #         ) 

    logger.log_pre_update(iter_counter,
    support_x,
    support_y,
    query_x,
    query_y,
    meta_learner,
    mode=mode)

    meta_learner.eval()
    # different inner-loop iter between train/test
    inner_iter = args.grad_steps_num_train if mode=='train' else args.grad_steps_num_eval
    # create graph only in the case of (train && second order)
    create_graph = (mode=='train') # and (not(args.first_order))
    for j in range(inner_iter):
        # get inner-grad
        in_pred = meta_learner(support_x, tuned_params)
        in_loss = F.cross_entropy(in_pred, support_y)
        in_grad = torch.autograd.grad(
            in_loss,
            tuned_params.values(),
            create_graph=create_graph
        )

        # update base-learner
        for k, g in zip(tuned_params.keys(), in_grad):
            tuned_params[k] = tuned_params[k] - args.lr_in * g
            
    meta_learner.train()
    # get outer-grad
    out_pred = meta_learner(query_x, tuned_params)
    out_loss = F.cross_entropy(out_pred, query_y)
    out_grad = torch.autograd.grad(
        out_loss,
        meta_learner.parameters()
    )
    
    # get computational graph
    # from torchviz import make_dot
    # img = make_dot(out_loss, params=dict(meta_learner.named_parameters()))
    # img.format = 'png'
    # img.render()
    # input()

    logger.log_post_update(iter_counter,
    support_x,
    support_y,
    query_x,
    query_y,
    meta_learner,
    tuned_params=tuned_params,
    mode=mode)

    return in_grad, out_grad

def outer_loop(args, meta_learner, opt, batch, logger, iter_counter):
    '''
    run a single batch == multiple episodes
    '''

    # move episode to device
    for i in range(len(batch)):
        batch[i] = batch[i].to(device)
    support_x, support_y, query_x, query_y = batch
    grad = [0. for p in meta_learner.parameters()]

    for i in range(args.batch_size):

        # accumulate grad to meta-learner using inner loop
        _, out_grad = inner_loop(args, 
        meta_learner, 
        support_x[i], 
        support_y[i],
        query_x[i],
        query_y[i],
        logger,
        iter_counter,
        mode='train')

        for j in range(len(out_grad)):
            grad[j] += out_grad[j]

    meta_learner.zero_grad()

    for p, g in zip(meta_learner.parameters(), grad):
        p.grad = g / float(args.batch_size)
        p.grad.data.clamp_(-10, 10)

    opt.step()

    # summarise inner loop and get validation performance
    logger.summarise_inner_loop(mode='train')

    return None

def train(args, meta_learner, opt, logger, path):

    sched = torch.optim.lr_scheduler.StepLR(opt, 5000, 0.9)

    # make datasets/ dataloaders
    # batchsz here means total episode number
    imagnet_dir = os.path.join(root_dir,"data/mini-imagenet")
    dataset_train = MetaMiniImageNet(imagnet_dir,split="train",batchsz=10000,n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, imsize=args.imgsz)
    dataset_valid = MetaMiniImageNet(imagnet_dir,split="val",batchsz=100,n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, imsize=args.imgsz)
    dataloader_train = DataLoader(dataset_train,
    batch_size=args.batch_size,
    shuffle=True, drop_last=True)
    dataloader_valid = DataLoader(dataset_valid,
    batch_size=args.batch_size,
    shuffle=True)


    iter_counter = 0
    while iter_counter < args.epoch:

        # iterate over epoch
        logger.print_header()

        for step, batch in enumerate(dataloader_train):

            logger.prepare_inner_loop(iter_counter)

            outer_loop(args, meta_learner, opt, batch, logger, iter_counter)

            # log/ save
            if (iter_counter % args.log_interval == 0):
                valid(args, meta_learner, dataloader_valid, logger,iter_counter, path)
                if path is not None:
                    np.save(path, [logger.training_stats, logger.validation_stats])
                    # save model to CPU
                    save_model = meta_learner
                    if device == 'cuda':
                        save_model = deepcopy(meta_learner).to('cpu')
                    torch.save(save_model, path)
            iter_counter += 1
            sched.step()

    return None

def valid(args, meta_learner, dataloader_valid, logger, iter_counter, path):
    logger.prepare_inner_loop(iter_counter, mode='valid')

    for step, batch in enumerate(dataloader_valid):

        # move episode to device
        for i in range(len(batch)):
            batch[i] = batch[i].to(device)
        support_x, support_y, query_x, query_y = batch

        for i in range(support_x.shape[0]):

            # accumulate grad to meta-learner using inner loop
            in_grad, out_grad = inner_loop(args, 
            meta_learner, 
            support_x[i], 
            support_y[i],
            query_x[i],
            query_y[i],
            logger,
            iter_counter,
            mode='valid')

    # this will take the mean over the batches
    logger.summarise_inner_loop(mode='valid')

    # keep track of best models
    logger.update_best_model(meta_learner, path)

    # print the log
    logger.print(iter_counter, in_grad, out_grad, mode='valid')

    return None


def main(args):

    set_seed(args.seed)

    # make nets
    # meta_learner = ResNet18(args.n_way).to(device)
    model_config = {"device": device, "growth_rate": 64, "n_block": 6, "n_channel": 64, "n_layer": 2,\
        "n_hidden": 256, "block_size":2 }
    meta_learner = Network(model_config, args.n_way).to(device)

    # make optimizer
    opt = torch.optim.Adam(meta_learner.parameters(), lr=args.lr_out)

    # make logger
    logger = Logger(args)

    # save path
    epochs = args.epoch    
    task_name = "{}_{}_{}way_{}shot_{}query_{}epoch".format("resnet18","MAML",args.n_way,args.k_spt,args.k_qry,epochs)
    summary = SummaryWriter(log_dir=os.path.join(log_dir,task_name))
    checkpoint_path = os.path.join(checkpoint_dir,task_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # train nets
    train(args, meta_learner, opt, logger, checkpoint_path)

    # write results
    return None

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, help='initial seed number', default=7)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--lr_out', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--lr_in', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--grad_steps_num_train', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--grad_steps_num_eval', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--batch_size', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--log_interval', type=int, help='interval of log', default=10)

    args = argparser.parse_args()

    main(args)
