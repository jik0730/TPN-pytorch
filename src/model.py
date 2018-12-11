import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

N_FILTERS = 64  # number of filters used in conv_block
N_NILTERS_R = 1  # number of filters used in last conv_block in relation module
K_SIZE = 3  # size of kernel
MP_SIZE = 2  # size of max pooling
# NOTE H_DIM should be small with ImageNet.. why?? MAYBE FIXED??
H_DIM = 8  # size of hidden layer of fc_layer used in relation module
EPS = 1e-8  # epsilon for numerical stability
TOP_K = 20  # k value for top-k extraction from W


class TPN(nn.Module):
    """
    Transductive Propagation Networks (TPN)
    Step 1. Feature embedding
    Step 2. Graph construction
    Step 3. Label propagation
    """

    def __init__(self, params):
        """
        Declare feature embedding and graph construction modules.
        Store data properties needed for defining the modules.
        Store model hyper-parameters.

        Args:
            in_channels: number of input channels (e.g. 3 for rgb)
            in_features_fc: number of input features for the first layer of
                            relation_module
            alpha: model hyper-paramter applied to computing label propagation
        """
        super(TPN, self).__init__()
        # Define data properties and model hyper-parameters
        self.params = params
        self.in_channels = params.in_channels
        self.in_features_fc = params.in_features_fc
        self.alpha = params.alpha

        # Define feature embedding networks.
        self.embed_mod = embedding_module(self.in_channels)

        # Define graph construction networks.
        self.rel_mod = relation_module(self.in_features_fc)

    def forward(self, X_sup, Y_sup, X_que, is_training=False):
        # Pass X_sup and X_que through feature embedding networks to be f(X_sup) and f(X_que).
        f_X_sup = self.embed_mod(X_sup)
        f_X_que = self.embed_mod(X_que)

        # Pass f(.)s to graph construction networks to get sigmas: g(.).
        g_f_X_sup = self.rel_mod(f_X_sup)
        g_f_X_que = self.rel_mod(f_X_que)

        # Construct similarity matrix (W) and propagate labels.
        F_star = self.label_prop(f_X_sup, f_X_que, g_f_X_sup, g_f_X_que, Y_sup)

        # Output predicted scores (labels) of X_que.
        # NOTE Refer to the review in openreview about reproduction.
        if is_training:
            return F_star
        else:
            return F_star[-X_que.size(0):, :]

    def label_prop(self, f_X_sup, f_X_que, g_f_X_sup, g_f_X_que, Y_sup):
        # Y_sup to one-hot format
        batch_size = Y_sup.size(0)
        num_classes = torch.max(Y_sup) + 1
        Y_sup_onehot = torch.Tensor(batch_size, num_classes)

        if self.params.cuda:
            Y_sup_onehot = Y_sup_onehot.cuda(async=True)

        Y_sup_onehot.zero_()
        Y_sup_onehot.scatter_(1, Y_sup.view(-1, 1), 1)

        # Build similarity matrix (W)
        f_X_sup = f_X_sup.view(f_X_sup.size(0), -1)
        f_X_que = f_X_que.view(f_X_que.size(0), -1)

        assert (
            f_X_sup.size(0) == g_f_X_sup.size(0)
        ), "Number of elements in minibatch should match: {} and {}".format(
            f_X_sup.size(0), g_f_X_sup.size(0))

        assert (g_f_X_sup.size(1) == 1
                ), "Output of relation module should be 1 but was {}".format(
                    g_f_X_sup.size(1))

        W_sup = f_X_sup / (g_f_X_sup + EPS)
        W_que = f_X_que / (g_f_X_que + EPS)
        # W_sup = f_X_sup / 1.
        # W_que = f_X_que / 1.
        W_cat = torch.cat([W_sup, W_que], dim=0)

        N_W_cat = W_cat.size(0)
        W1 = W_cat.unsqueeze(0).repeat(N_W_cat, 1, 1)
        W2 = W_cat.unsqueeze(1).repeat(1, N_W_cat, 1)
        dist = torch.mean((W1 - W2)**2, dim=2)

        W = torch.exp(-dist / 2.)
        W = self.top_k(W)  # NOTE why we use top-k NN graph ???

        assert (W.size(0) == (W_sup.size(0) + W_que.size(0)))

        # Compute graph Laplacians (S) NOTE this is unstable (BE CAREFUL)
        # D = torch.sum(W, dim=1)
        # D = torch.diag(D)
        # D_inv = torch.inverse(D + EPS * torch.eye(D.size(0)))
        # D_sqrt_inv = torch.sqrt(D_inv)
        D = torch.sum(W, dim=1)
        D_inv = 1.0 / (D + EPS)
        D_sqrt_inv = torch.sqrt(D_inv)
        D_size = D_sqrt_inv.size(0)
        S = D_sqrt_inv.repeat(D_size, 1) * W * D_sqrt_inv.view(-1, 1).repeat(
            1, D_size)

        # Label propagation (F*)
        Y_que = torch.zeros([f_X_que.size(0), Y_sup_onehot.size(1)]) + EPS
        identity_for_F_star = torch.eye(S.size(0))

        if self.params.cuda:
            Y_que = Y_que.cuda(async=True)
            identity_for_F_star = identity_for_F_star.cuda(async=True)

        Y = torch.cat([Y_sup_onehot, Y_que], dim=0)
        F_star = torch.inverse(identity_for_F_star - self.alpha * S + EPS)
        F_star = torch.matmul(F_star, Y.to(torch.float))
        F_star_softmax = F.log_softmax(F_star, dim=1)

        return F_star_softmax

    def top_k(self, W, k=TOP_K):
        """
        Return the matrix of top k values of each row of W.
        Otherwise, 0.
        """
        val, ind = torch.topk(W, k)
        W_top_k_ind = torch.zeros(W.size(), device=W.device)
        W_top_k_ind = W_top_k_ind.scatter_(1, ind, 1)
        W_top_k_ind = (W_top_k_ind > 0) | (W_top_k_ind.t() > 0)
        W_top_k = W * W_top_k_ind.to(W.dtype)
        return W_top_k


class embedding_module(nn.Module):
    """
    The embedding module embeds inputs (features) in embedding space, which 
    learns meta-information that is good for classification.
    Architecture of the module is fixed (following related recent works):
        (conv_block followed by 2x2 max-pool) * 2
        (conv_block) * 2
    """

    def __init__(self, in_channels):
        """
        Args:
            in_channels: number of input channels feeding into first conv_block
        """
        super(embedding_module, self).__init__()
        self.embed_mod = nn.Sequential(
            conv_block(in_channels, padding=1, pooling=True),
            conv_block(N_FILTERS, padding=1, pooling=True),
            conv_block(N_FILTERS, padding=1, pooling=True),
            conv_block(N_FILTERS, padding=1, pooling=True))

    def forward(self, X):
        """
        Returns:
            [N, 64, 1, 1] for Omniglot (28x28)
            [N, 64, 5, 5] for miniImageNet (84x84)
        """
        return self.embed_mod(X)


class relation_module(nn.Module):
    """
    The embedding module meta-learns example-wise length scale parameter in 
    similarity matrix.
    Architecture of the module is fixed (following related recent works):
        (conv_block followed by 2x2 max-pool) * 2
        (fc_layer) * 2
    """

    def __init__(self, in_features_fc):
        """
        Args:
            in_features_fc: number of input features for the first fc_layer
                1 for Omniglot and 1 for miniImageNet given current settings
        """
        super(relation_module, self).__init__()
        # NOTE hard-coding for implementation issue
        # if omniglot, then padding=2
        # if imagenet, then padding=1
        # if dataset == 'Omniglot':
        #     pad = 2
        # elif dataset == 'ImageNet':
        #     pad = 1
        self.rel_mod = nn.Sequential(
            conv_block(N_FILTERS, padding=1, pooling=True, pool_pad=1),  # NOTE
            conv_block(
                N_FILTERS, N_NILTERS_R, padding=1, pooling=True, pool_pad=1))
        self.fc1 = nn.Linear(in_features_fc, H_DIM)
        self.fc2 = nn.Linear(H_DIM, 1)

    def forward(self, X):
        """
        Args:
            X: [1, 64, 1, 1] for Omniglot (28x28)
               [1, 64, 5, 5] for miniImageNet (84x84)
        Returns:
            out: Nx1 sigmas
        """
        out = self.rel_mod(X)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


def conv_block(in_channels,
               out_channels=N_FILTERS,
               padding=0,
               pooling=True,
               pool_pad=0):
    """
    The unit architecture (Convolutional Block; CB) used in the modules.
    The CB consists of following modules in the order:
        3x3 conv, 64 filters
        batch normalization
        ReLU
        MaxPool
    """
    if pooling:
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, K_SIZE, padding=padding),
            nn.BatchNorm2d(out_channels, momentum=1, affine=True), nn.ReLU(),
            nn.MaxPool2d(MP_SIZE, padding=pool_pad))
    else:
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, K_SIZE, padding=padding),
            nn.BatchNorm2d(out_channels, momentum=1, affine=True), nn.ReLU())
    return conv


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(labels.size)


# Maintain all metrics required in this dictionary.
# These are used in the training and evaluation loops.
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}