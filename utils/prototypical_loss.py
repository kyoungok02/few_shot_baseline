import torch
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.autograd import Variable
from utils.train_utils import AverageMeter

# from https://github.com/oscarknagg/few-shot
def pairwise_distances(x, y, matching_fn):
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.
    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0] # n_query
    n_y = y.shape[0] # n_class

    # [n_class, n_query, dim]형태로 변경
    if matching_fn.lower() == 'l2' or matching_fn.lower == 'euclidean':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    # if matching_fn.lower() == 'l2' or matching_fn.lower == 'euclidean':
    #     distances = (
    #             x.unsqueeze(0).expand(n_y, n_x, -1) -
    #             y.unsqueeze(1).expand(n_y, n_x, -1)
    #     ).pow(2).sum(dim=2)
    #     return distances
    # 변경 안함
    elif matching_fn.lower() == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn.lower() == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise (ValueError('Unsupported similarity function'))


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''

    def __init__(self):
        super(PrototypicalLoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self,input_sup, input_query, target_sup, target_query, device):
        return prototypical_loss(input_sup, input_query, target_sup, target_query, device)


def prototypical_loss(input_sup, target_sup, input_query, target_query, device):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
    '''
    # batch가 고려되어 있지 않는 경우 임
    # batch를 최후에 더해주거나 해줘야 함
    batch_size = input_sup.size(0)
    classes = torch.unique(target_sup)
    n_class = len(classes)
    n_query = int(input_query.size(1) / n_class) # query 갯수 
    acc_val = AverageMeter()
    batch_loss = []
    for task in range(batch_size):
        s_in, s_target, q_in, q_target = input_sup[task,:,:], target_sup[task,:], input_query[task,:,:], target_query[task,:]
        support_idxs = torch.stack(list(map(lambda c: s_target.eq(c).nonzero().squeeze(1), classes)))
        # [class, n_support, dim] 형태로 변경
        # torch.stack([torch.stack([input_sup[idx[0].item(),idx[1].item(),:] for idx in idx_list]) for idx_list in support_idxs])
        # n_support에 대해서 mean을 취해줌
        # [class, dim]형태로 나옴
        prototypes = torch.stack([torch.stack([s_in[idx.item(),:] for idx in idx_list]).mean(0) for idx_list in support_idxs])

        # Make query samples
        # [n_class * n_query, dim]형태로 나옴
        query_idxs = torch.stack(list(map(lambda c: q_target.eq(c).nonzero(), classes))).view(-1)
        query_samples = torch.stack([q_in[idx.item(), :] for idx in query_idxs])

        dists = pairwise_distances(query_samples, prototypes, 'l2')

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
        _, y_hat = log_p_y.max(2)
        
        # target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        # target_inds = Variable(target_inds, requires_grad=False)
        target_label = torch.arange(0, n_class).view(n_class,1, 1).expand(n_class,n_query, 1).long().to(device)
        target_label = Variable(target_label, requires_grad = False)

        loss = -log_p_y.gather(2, target_label).squeeze().view(-1).mean()
        # loss_val = torch.nn.NLLLoss()(log_p_y, target_label)
        acc = y_hat.eq(target_label.squeeze()).float().mean()
        batch_loss.append(loss)
        acc_val.update(acc.item())
    loss_val = torch.stack([loss_v for loss_v in batch_loss])

    return loss_val.mean(0), { 'loss': loss_val.mean(0).item(), 'acc': acc_val.val}