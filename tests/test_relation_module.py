import unittest
import torch
import torch.nn as nn
from src.model import embedding_module
from src.model import relation_module


class test_relation_module(unittest.TestCase):
    def setUp(self):
        pass

    def test_input_dim_Omniglot(self):
        in_channels = 1
        X = torch.ones([1, in_channels, 28, 28])
        em = embedding_module(in_channels)
        rm = relation_module(1)
        f_X = em(X)
        g_f_X = rm(f_X)
        self.assertTupleEqual(g_f_X.size(), (1, 1))

    def test_input_dim_miniImageNet(self):
        in_channels = 3
        X = torch.ones([1, in_channels, 84, 84])
        em = embedding_module(in_channels)
        rm = relation_module(4)
        f_X = em(X)
        g_f_X = rm(f_X)
        self.assertTupleEqual(g_f_X.size(), (1, 1))

    def test_architecture(self):
        in_channels = 3
        X = torch.rand([2, in_channels, 84, 84])
        Y = torch.ones([2, 1])
        Y[0, 0] = 0
        em = embedding_module(in_channels)
        rm = relation_module(4)
        before_params = [p.clone() for p in rm.parameters()]

        optimizer = torch.optim.Adam(rm.parameters())
        loss_fn = nn.BCEWithLogitsLoss()

        f_X = em(X)
        g_f_X = rm(f_X)
        loss = loss_fn(g_f_X, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        after_params = [p.clone() for p in rm.parameters()]

        for b_param, a_param in zip(before_params, after_params):
            # Make sure something changed.
            # print(b_param)
            # print(a_param)
            self.assertTrue((b_param != a_param).any())


if __name__ == '__main__':
    unittest.main()