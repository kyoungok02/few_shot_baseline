import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class Network(nn.Module):
    def __init__(self,config, n_way):
        super().__init__()
        self.device = config["device"]
        self.n_block = config["n_block"]
        self.n_channel = config["n_channel"]
        self.n_layer = config["n_layer"]
        self.layers = nn.ParameterDict(OrderedDict([]))
        # add convolution blocks
        for i in range(self.n_block):
            in_channel = 3 if i==0 else self.n_channel
            self.layers.update(
                OrderedDict([
                    ('conv_{}_weight'.format(i), nn.Parameter(torch.zeros(self.n_channel, in_channel, 3, 3))),
                    ('conv_{}_bias'.format(i), nn.Parameter(torch.zeros(self.n_channel))),
                    ('bn_{}_weight'.format(i), nn.Parameter(torch.zeros(self.n_channel))),
                    ('bn_{}_bias'.format(i), nn.Parameter(torch.zeros(self.n_channel))),
                ])
            )
        # add fc layers (note that this architecture is different from original MAML)
        for i in range(self.n_layer):
            in_size = self.n_channel * 5 * 5 if i==0 else config["n_hidden"]
            out_size = n_way if i==(self.n_layer-1) else config["n_hidden"]
            self.layers.update(
                OrderedDict([
                    ('fc_{}_weight'.format(i), nn.Parameter(torch.zeros(out_size, in_size))),
                    ('fc_{}_bias'.format(i), nn.Parameter(torch.zeros(out_size)))
                ])
            )

        self.init_params()

    def init_params(self):

        for k, v in self.named_parameters():
            if ('conv' in k) or ('fc' in k):
                if ('weight' in k):
                    nn.init.kaiming_normal_(v)
                elif ('bias' in k):
                    nn.init.constant_(v, 0.0)
            elif ('bn' in k):
                if ('weight' in k):
                    nn.init.constant_(v, 1.0)
                elif ('bias' in k):
                    nn.init.constant_(v, 0.0)

    def forward(self, x, tuned_params=None):

        if tuned_params is None:
            params = OrderedDict([(k,v) for k,v in self.named_parameters()])
        else:
            params = OrderedDict([])
            for k, v in self.named_parameters():
                if k in tuned_params.keys():
                    params[k] = tuned_params[k]
                else:
                    params[k] = v

        for i in range(self.n_block):
            x = F.conv2d(x,
            weight=params['layers.conv_{}_weight'.format(i)],
            bias=params['layers.conv_{}_bias'.format(i)],
            padding=1)
            x = F.batch_norm(x,
            running_mean=None,
            running_var=None,
            weight=params['layers.bn_{}_weight'.format(i)],
            bias=params['layers.bn_{}_bias'.format(i)],
            training=True)
            x = F.leaky_relu(x, 0.1)
            if i < 4:
                x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, self.n_channel * 5 * 5)

        for i in range(self.n_layer):
            x = F.linear(x,
            weight=params['layers.fc_{}_weight'.format(i)],
            bias=params['layers.fc_{}_bias'.format(i)])
            if (i < self.n_layer-1):
                x = F.leaky_relu(x, 0.1)
        
        return x

class DenseNet(nn.Module):
    def __init__(self, config, n_way):
        super().__init__()
        self.device = config["device"]
        self.growth_rate = config["growth_rate"]
        self.n_block = config["n_block"]
        self.block_size = config["block_size"]
        self.n_layer = config["n_layer"]
        self.layers = nn.ParameterDict(OrderedDict([]))
        self.drop_out = nn.Dropout(p=0.1)

        # add init conv block
        self.layers.update(
                OrderedDict([
                    ('bn_weight', nn.Parameter(torch.zeros(3))),
                    ('bn_bias', nn.Parameter(torch.zeros(3))),
                    ('conv_weight', nn.Parameter(torch.zeros(config["n_channel"], 3, 3, 3))),
                    ('conv_bias', nn.Parameter(torch.zeros(config["n_channel"]))),
                ])
            )
        
        # add dense blocks
        start_filter = config["n_channel"]
        for i in range(self.n_block-1):
            for j in range(self.block_size):
                self.layers.update(OrderedDict([
                    ('bn_bottleneck_{}_{}_weight'.format(i,j), 
                    nn.Parameter(torch.zeros(self.growth_rate*j + start_filter))),
                    ('bn_bottleneck_{}_{}_bias'.format(i,j), 
                    nn.Parameter(torch.zeros(self.growth_rate*j + start_filter))),
                    ('conv_bottleneck_{}_{}_weight'.format(i,j), 
                    nn.Parameter(torch.zeros(self.growth_rate, self.growth_rate*j + start_filter, 1, 1))),
                    ('conv_bottleneck_{}_{}_bias'.format(i,j), 
                    nn.Parameter(torch.zeros(self.growth_rate))),
                    ('bn_{}_{}_weight'.format(i,j), 
                    nn.Parameter(torch.zeros(self.growth_rate))),
                    ('bn_{}_{}_bias'.format(i,j), 
                    nn.Parameter(torch.zeros(self.growth_rate))),
                    ('conv_{}_{}_weight'.format(i,j), 
                    nn.Parameter(torch.zeros(self.growth_rate, self.growth_rate, 3, 3))),
                    ('conv_{}_{}_bias'.format(i,j), 
                    nn.Parameter(torch.zeros(self.growth_rate))),
                ]))
            self.layers.update(OrderedDict([
                ('bn_transition_{}_weight'.format(i), 
                nn.Parameter(torch.zeros(self.growth_rate*self.block_size + start_filter))),
                ('bn_transition_{}_bias'.format(i), 
                nn.Parameter(torch.zeros(self.growth_rate*self.block_size + start_filter))),
                ('conv_transition_{}_weight'.format(i),
                nn.Parameter(torch.zeros(int(0.5*(self.growth_rate*self.block_size + start_filter)) ,self.growth_rate*self.block_size + start_filter, 1, 1))),
                ('conv_transition_{}_bias'.format(i),
                nn.Parameter(torch.zeros(int(0.5*(self.growth_rate*self.block_size + start_filter)) ))),
            ]))
            start_filter = int(0.5*(self.growth_rate*self.block_size + start_filter))

        # add fc layers
        for i in range(self.n_layer):
            in_size = start_filter * 5 * 5 if i==0 else config["n_hidden"]
            out_size = n_way if i==(self.n_layer-1) else config["n_hidden"]
            self.layers.update(OrderedDict([
                ('fc_{}_weight'.format(i), nn.Parameter(torch.zeros(out_size, in_size))),
                ('fc_{}_bias'.format(i), nn.Parameter(torch.zeros(out_size)))
            ]))

        self.init_params()

    def init_params(self):

        for k, v in self.named_parameters():
            if ('conv' in k) or ('fc' in k):
                if ('weight' in k):
                    nn.init.kaiming_normal_(v)
                elif ('bias' in k):
                    nn.init.constant_(v, 0.0)
            elif ('bn' in k):
                if ('weight' in k):
                    nn.init.constant_(v, 1.0)
                elif ('bias' in k):
                    nn.init.constant_(v, 0.0)

    def forward(self, x, tuned_params=None):

        if tuned_params is None:
            params = OrderedDict([(k,v) for k,v in self.named_parameters()])
        else:
            params = OrderedDict([])
            for k, v in self.named_parameters():
                if k in tuned_params.keys():
                    params[k] = tuned_params[k]
                else:
                    params[k] = v

        # apply init conv block
        x = F.batch_norm(x,
        running_mean=None,
        running_var=None,
        weight=params['layers.bn_weight'],
        bias=params['layers.bn_bias'],
        training=True)
        x = F.leaky_relu(x, 0.1)
        x = F.conv2d(x,
        weight=params['layers.conv_weight'],
        bias=params['layers.conv_bias'],
        padding=1)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # apply dense blocks
        for i in range(self.n_block-1):
            for j in range(self.block_size):
                # apply bottleneck conv
                x_cur = F.batch_norm(x,
                running_mean=None,
                running_var=None,
                weight=params['layers.bn_bottleneck_{}_{}_weight'.format(i,j)],
                bias=params['layers.bn_bottleneck_{}_{}_bias'.format(i,j)],
                training=True)
                x_cur = F.leaky_relu(x_cur, 0.1)
                x_cur = F.conv2d(x_cur,
                weight=params['layers.conv_bottleneck_{}_{}_weight'.format(i,j)],
                bias=params['layers.conv_bottleneck_{}_{}_bias'.format(i,j)])
                # apply conv
                x_cur = F.batch_norm(x_cur,
                running_mean=None,
                running_var=None,
                weight=params['layers.bn_{}_{}_weight'.format(i,j)],
                bias=params['layers.bn_{}_{}_bias'.format(i,j)],
                training=True)
                x_cur = F.leaky_relu(x_cur, 0.1)
                x_cur = F.conv2d(x_cur,
                weight=params['layers.conv_{}_{}_weight'.format(i,j)],
                bias=params['layers.conv_{}_{}_bias'.format(i,j)],
                padding=1)
                x_cur = self.drop_out(x_cur)
                x = torch.cat((x, x_cur), 1)

            # apply transition conv
            x = F.batch_norm(x,
            running_mean=None,
            running_var=None,
            weight=params['layers.bn_transition_{}_weight'.format(i)],
            bias=params['layers.bn_transition_{}_bias'.format(i)],
            training=True)
            x = F.leaky_relu(x, 0.1)
            x = F.conv2d(x,
            weight=params['layers.conv_transition_{}_weight'.format(i)],
            bias=params['layers.conv_transition_{}_bias'.format(i)])
            if i < 3:
                x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, x.shape[1] * 5 * 5)

        for i in range(self.n_layer):
            x = F.leaky_relu(x, 0.1)
            x = F.linear(x,
            weight=params['layers.fc_{}_weight'.format(i)],
            bias=params['layers.fc_{}_bias'.format(i)])
        
        return x
