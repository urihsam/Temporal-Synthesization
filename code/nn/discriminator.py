import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils.train_utils import to_var

class CNN_Discriminator(nn.Module):

    def __init__(self, feature_size, feature_dropout, filter_size, window_sizes):
        super().__init__()
        self.feature_size = feature_size
        self.feature_dropout = nn.Dropout(p=feature_dropout)

        self.filter_size = filter_size
        self.window_sizes = window_sizes

        self.convs = nn.ModuleList([torch.nn.Conv2d(1, filter_size, (window_size, self.feature_size), padding=(1, 0)) for window_size in window_sizes])
        self.feature2binary = nn.Linear(filter_size * len(window_sizes), 1)

    def forward(self, prob_sequence, input_mask=None):
        batch_size = prob_sequence.size(0)

        input_ = prob_sequence.view(batch_size, -1, self.feature_size)
        if input_mask != None:
            input_ = torch.mul(input_, input_mask)

        input_ = input_.unsqueeze(1).float() # num_channel
        feature_maps = [nn.functional.leaky_relu(conv(input_), 0.2).squeeze(3) for conv in self.convs]
        feature_maps = [nn.functional.max_pool1d(feature_map, feature_map.size(2)).squeeze(2) for feature_map in feature_maps]

        feature_vecs = torch.cat(feature_maps, 1)
        feature_vecs = self.feature_dropout(feature_vecs)
        scores = self.feature2binary(feature_vecs).squeeze(1)
        return scores

    def cal_gradient_penalty(self, real_data, fake_data, input_mask=None, gp_lambda=10):
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1)
        epsilon = epsilon.expand(real_data.size())

        if torch.cuda.is_available():
            epsilon = epsilon.cuda()

        interpolates = epsilon * real_data + (1 - epsilon) * fake_data
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        D_interpolates = self.forward(interpolates, input_mask)

        gradients = torch.autograd.grad(outputs=D_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(D_interpolates.size()).cuda() if torch.cuda.is_available() else torch.ones(D_interpolates.size()),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
        return gradient_penalty