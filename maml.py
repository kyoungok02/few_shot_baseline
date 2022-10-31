import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    copy import deepcopy

class ProtoMAML(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, encoder):
        """

        :param args:
        """
        super(ProtoMAML, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.net = encoder 
        self.optimizers = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizers, milestones=[140,180], gamma=0.1)
        
    def set_parameter_requires_grad(self, model, index):
        for param in model.parameters():
            param.requires_grad = index
            
    def run_model(self, model, output_weight, output_bias, img, labels):
        feats = model(img)
        preds = F.linear(feats, output_weight, output_bias)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=1) == labels).float()
        return loss, preds, acc

    def calculate_prototypes(self, features, targets):
        # Given a stack of features vectors and labels, return class prototypes
        # features - shape [N, proto_dim], targets - shape [N]
        classes, _ = torch.unique(targets).sort()  # Determine which classes we have
        prototypes = []
        for c in classes:
            p = features[torch.where(targets == c)[0]].mean(dim=0)  # Average class feature vectors
            prototypes.append(p)
        prototypes = torch.stack(prototypes, dim=0)
        # Return the 'classes' tensor to know which prototype belongs to which class
        return prototypes, classes
    
    def inner_loop(self, x_spt, y_spt):
        '''
        train the base model using samples
        '''
        # Create inner-loop model and optimizer
        local_model = deepcopy(self.net)
        local_model.train()
        local_optim = optim.SGD(local_model.parameters(), lr=self.update_lr)
        local_optim.zero_grad()
        # determine prototype initialization
        support_feats = self.net(x_spt)
        # Create output layer weights with prototype-based initialization
        prototypes, classes = self.calculate_prototypes(support_feats, y_spt)
        init_weight = 2 * prototypes
        init_bias = -torch.norm(prototypes, dim=1)**2
        output_weight = init_weight.detach().requires_grad_()
        output_bias = init_bias.detach().requires_grad_()


        # Optimize inner loop model on support set
        for _ in range(self.update_step):
            # Determine loss on the support set
            with torch.enable_grad():
                loss, _, _ = self.run_model(local_model, output_weight, output_bias, x_spt, y_spt)
                # Calculate gradients and perform inner loop update
                loss.backward()
                local_optim.step()
                # Update output layer via SGD
                output_weight.data -= self.meta_lr * output_weight.grad
                output_bias.data -= self.meta_lr * output_bias.grad
                # Reset gradients
                local_optim.zero_grad()
                output_weight.grad.fill_(0)
                output_bias.grad.fill_(0)

        # Re-attach computation graph of prototypes
        output_weight = (output_weight - init_weight).detach() + init_weight
        output_bias = (output_bias - init_bias).detach() + init_bias

        return local_model, output_weight, output_bias, classes

# dataloader 안되서 batch 제외
    def forward(self, x_spt, y_spt, x_qry, y_qry, mode="train"):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        accuracies = []
        losses = []
        self.optimizers.zero_grad()
        
        if mode == "train":
            self.set_parameter_requires_grad(self.net, True)
        
        # inner-loop
        local_model, output_weight, output_bias, classes = self.inner_loop(x_spt, y_spt)
        # Determine loss of query set
        query_labels = (classes[None,:] == y_qry[:,None]).long().argmax(dim=-1)
        loss, preds, acc = self.run_model(local_model, output_weight, output_bias, x_qry, query_labels)
        # Calculate gradients for query set loss
        if mode == "train":
            loss.backward()
            for p_global, p_local in zip(self.net.parameters(), local_model.parameters()):
                p_global.grad += p_local.grad  # First-order approx. -> add gradients of finetuned and base model

        accuracies.append(acc.mean().detach())
        losses.append(loss.detach())

        # Perform update of base model
        if mode == "train":
            self.optimizers.step()
            self.set_parameter_requires_grad(self.net, False)

        # print(f"{mode}_loss", sum(losses) / len(losses))
        # print(f"{mode}_acc", sum(accuracies) / len(accuracies))
        return (sum(losses) / len(losses)), (sum(accuracies) / len(accuracies))
    


class MAML(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, encoder):
        """

        :param args:
        """
        super(MAML, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.net = encoder 
        self.optimizers = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizers, milestones=[140,180], gamma=0.1)
        
    def set_parameter_requires_grad(self, model, index):
        for param in model.parameters():
            param.requires_grad = index
            
    def run_model(self, model, output_weight, output_bias, img, labels):
        feats = model(img)
        preds = F.linear(feats, output_weight, output_bias)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=1) == labels).float()
        return loss, preds, acc

    
    def inner_loop(self, x_spt, y_spt):
        '''
        train the base model using samples
        '''
        # Create inner-loop model and optimizer
        local_model = deepcopy(self.net)
        local_model.train()
        local_optim = optim.SGD(local_model.parameters(), lr=self.update_lr)
        local_optim.zero_grad()
        # determine prototype initialization
        support_feature = self.net(x_spt)
        # Create output layer weights with prototype-based initialization
        classes, _ = torch.unique(y_spt).sort()  # Determine which classes we have
        init_weight = torch.ones(classes.size(0), support_feature.size(1))
        init_bias = -torch.norm(init_weight, dim=1)
        output_weight = init_weight.detach().requires_grad_()
        output_bias = init_bias.detach().requires_grad_()
        # Optimize inner loop model on support set
        for _ in range(self.update_step):
            # Determine loss on the support set
            with torch.enable_grad():
                loss, _, _ = self.run_model(local_model, output_weight, output_bias, x_spt, y_spt)
                loss.backward(retain_graph=True)
                local_optim.step()
                # Calculate gradients and perform inner loop update
                grad_weigt = torch.autograd.grad(loss,output_weight)
                output_weight = list(map(lambda p:p[1] - self.update_lr*p[0], zip(grad_weigt, output_weight)))[0]
                output_bias = torch.norm(output_weight, dim=1)
                local_optim.zero_grad()


        return local_model, output_weight, output_bias, classes


# dataloader 안되서 batch 제외
    def forward(self, x_spt, y_spt, x_qry, y_qry, mode="train"):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        accuracies = []
        losses = []
        self.optimizers.zero_grad()
        
        #batch 생략        
        if mode == "train":
            self.set_parameter_requires_grad(self.net, True)
        
        # inner-loop
        local_model, output_weight, output_bias, classes = self.inner_loop(x_spt, y_spt)
        # Determine loss of query set
        query_labels = (classes[None,:] == y_qry[:,None]).long().argmax(dim=-1)
        loss, _, acc = self.run_model(local_model, output_weight, output_bias, x_qry, query_labels)
        # Calculate gradients for query set loss
        if mode == "train":
            # Calculate gradients and perform inner loop update
            grads = torch.autograd.grad(loss,output_weight)
            output_weight = list(map(lambda p:p[1] - self.meta_lr*p[0], zip(grads, output_weight)))[0]
            output_bias = torch.norm(output_weight, dim=1)
            loss, _, acc = self.run_model(local_model, output_weight, output_bias, x_qry, query_labels)
            loss.backward()            
        accuracies.append(acc.mean().detach())
        losses.append(loss.detach())

        # Perform update of base model
        if mode == "train":
            self.optimizers.step()
            self.set_parameter_requires_grad(self.net, False)

        return (sum(losses) / len(losses)), (sum(accuracies) / len(accuracies))