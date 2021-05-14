import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils.train_utils import to_var

class MLP_Generator(nn.Module):
    def __init__(self, input_size, output_size, archs, activation=nn.LeakyReLU(0.2)):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        layer_sizes = [input_size] + archs
        layers = []

        fc_layers = []
        bn_layers = []
        ac_layers = []

        for i in range(len(layer_sizes)-1):
            fc_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=False))
            bn_layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            ac_layers.append(activation)

        fc_layers.append(nn.Linear(layer_sizes[-1], output_size, bias=False))
        bn_layers.append(nn.BatchNorm1d(output_size))
        ac_layers.append(nn.Tanh())

        self.fc_layers = nn.ModuleList(fc_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        self.ac_layers = nn.ModuleList(ac_layers)

    def forward(self, x=None, batch_size=None):
        if x is None:
            x = to_var(torch.randn([batch_size, self.input_size]))
        
        for i in range(len(self.fc_layers)):
            x1 = self.fc_layers[i](x)
            x2 = self.bn_layers[i](x1)
            x3 = self.ac_layers[i](x2)
            if i != len(self.fc_layers)-1:
                x = x + x3
            else:
                x = x3
        
        return x

class MLP_Discriminator(nn.Module):
    def __init__(self, input_size, output_size, archs, activation=nn.LeakyReLU(0.2)):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        layer_sizes = [input_size] + archs
        layers = []

        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(activation)

        layers.append(nn.Linear(layer_sizes[-1], output_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.cat((x, x.mean(dim=0).repeat([batch_size, 1])), dim=1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x.squeeze(1)

    def cal_gradient_penalty(self, real_data, fake_data, gp_lambda=10):
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1)
        epsilon = epsilon.expand(real_data.size())

        if torch.cuda.is_available():
            epsilon = epsilon.cuda()

        interpolates = epsilon * real_data + (1 - epsilon) * fake_data
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        D_interpolates = self.forward(interpolates)

        gradients = torch.autograd.grad(outputs=D_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(D_interpolates.size()).cuda() if torch.cuda.is_available() else torch.ones(D_interpolates.size()),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
        return gradient_penalty

