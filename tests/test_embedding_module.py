import unittest
import torch
import torch.nn as nn
from src.model import embedding_module


class test_embedding_module(unittest.TestCase):
    def setUp(self):
        pass

    def test_input_dim_Omniglot(self):
        in_channels = 1
        X = torch.ones([1, in_channels, 28, 28])
        em = embedding_module(in_channels)
        f_X = em(X)
        self.assertTupleEqual(f_X.size(), (1, 64, 1, 1))

    def test_input_dim_miniImageNet(self):
        in_channels = 3
        X = torch.ones([1, in_channels, 84, 84])
        em = embedding_module(in_channels)
        f_X = em(X)
        self.assertTupleEqual(f_X.size(), (1, 64, 5, 5))

    def test_architecture(self):
        in_channels = 3
        X = torch.ones([1, in_channels, 84, 84])
        Y = torch.ones([1, 64, 5, 5])
        em = embedding_module(in_channels)
        before_params = [p.clone() for p in em.parameters()]

        optimizer = torch.optim.Adam(em.parameters())
        loss_fn = nn.BCEWithLogitsLoss()

        f_X = em(X)
        loss = loss_fn(f_X, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        after_params = [p.clone() for p in em.parameters()]

        for b_param, a_param in zip(before_params, after_params):
            # Make sure something changed.
            self.assertTrue((b_param != a_param).any())


if __name__ == '__main__':
    unittest.main()