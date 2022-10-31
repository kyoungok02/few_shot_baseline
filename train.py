from tabnanny import check
import  torch, os
import  numpy as np
from torchinfo import summary
from    utils.MiniImagenet import MetaMiniImageNet
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
import shutil

from utils.Models import initialize_model
from maml import ProtoMAML,MAML

from tensorboardX import SummaryWriter

root_dir = "/Users/kyoung-okyang/few_shot/few_shot_baseline"
# device = torch.device('mps' if torch.has_mps else 'cpu')
checkpoint_dir = root_dir + "/checkpoint"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
log_dir = root_dir + "/logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
device=torch.device('cpu')


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h



def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    epochs = args.epoch    
    task_name = "{}_{}_{}way_{}shot_{}query_{}epoch".format("resnet18","MAML",args.n_way,args.k_spt,args.k_qry,epochs)
    summary = SummaryWriter(log_dir=os.path.join(log_dir,task_name))
    checkpoint_path = os.path.join(checkpoint_dir,task_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        
    print(args)
    
    # encoder, input_size = initialize_model("resnet18",args.n_way, feature_extract=True, use_pretrained=True)
    encoder, input_size = initialize_model("resnet18",256, feature_extract=True, use_pretrained=True)
    # maml = ProtoMAML(args, encoder).to(device)
    maml = MAML(args, encoder).to(device)
    
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    imagnet_dir = os.path.join(root_dir,"data/mini-imagenet")
    mini = MetaMiniImageNet(imagnet_dir,split="train",batchsz=10000,n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, imsize=args.imgsz)
    mini_val = MetaMiniImageNet(imagnet_dir,split="val",batchsz=100,n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, imsize=args.imgsz)
    
    best_acc = 0.0

    for epoch in range(epochs):
        # fetch meta_batchsz num of episode each time
        # db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)
        maml.train()
        print("Training Epoch[{}/{}]".format(epoch,epochs))
        # for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(mini):
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            tr_loss, tr_accs = maml(x_spt, y_spt, x_qry, y_qry,"train")
            if step % 30 == 0:
                print('step:', step, '\ttraining acc:', tr_accs)
            if step % 500 == 0:  # evaluation
                print("Validation ================================================================================")
                # db_test = DataLoader(mini_val, 1, shuffle=True, num_workers=1, pin_memory=True)
                maml.eval()
                accs_all_test = []
                # for x_spt, y_spt, x_qry, y_qry in db_test:
                for x_spt, y_spt, x_qry, y_qry in mini_val:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    val_loss, val_accs = maml(x_spt, y_spt, x_qry, y_qry,"val")
                    accs_all_test.append(val_accs)
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                is_best = accs > best_acc
                save_checkpoint(maml, checkpoint_path, is_best)
                print('Validation acc:', accs)
        summary.add_scalar('loss/loss', {"train_losses": tr_loss.item(), "val_losses": val_loss.item()}, epoch)
        summary.add_scalar('accuracy/accuracies', {"train_accuracy": tr_accs.item(), "val_accuracy": val_accs.item()}, epoch)

    print("Training Completed!!")

def save_checkpoint(model, path, is_best):
    filename = '%s/checkpoint.pt' % (path)
    torch.save(model.state_dict(), filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('checkpoint.pt', 'best.pt'))

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main(args)