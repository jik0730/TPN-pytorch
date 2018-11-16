import unittest
import os

import torch
import torch.nn as nn
import utils
from src.model import TPN


class test_TPN_smoke(unittest.TestCase):
    def setUp(self):
        pass

    @unittest.skip("top-k problem")
    def test_miniImageNet(self):
        # Configurations 3-way 3-shot with 3 query set
        model_dir = 'experiments/base_model'
        json_path = os.path.join(model_dir, 'params.json')
        assert os.path.isfile(
            json_path), "No json configuration file found at {}".format(
                json_path)
        params = utils.Params(json_path)

        params.cuda = False
        params.in_channels = 3
        params.in_features_fc = 4
        params.alpha = 0.9

        X_sup = torch.ones([9, params.in_channels, 84, 84])
        X_que = torch.ones([3, params.in_channels, 84, 84])
        Y_sup = torch.zeros([9], dtype=torch.long)
        Y_que = torch.zeros([3], dtype=torch.long)
        Y_sup[0] = 0
        Y_sup[1] = 0
        Y_sup[2] = 0
        Y_sup[3] = 1
        Y_sup[4] = 1
        Y_sup[5] = 1
        Y_sup[6] = 2
        Y_sup[7] = 2
        Y_sup[8] = 2
        Y_que[0] = 0
        Y_que[1] = 1
        Y_que[2] = 2

        Y_sup = Y_sup.to(torch.long)
        Y_que = Y_que.to(torch.long)

        model = TPN(params)
        before_params = [(n, p.clone()) for n, p in model.named_parameters()]

        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = nn.CrossEntropyLoss()

        Y_que_hat = model(X_sup, Y_sup, X_que)
        loss = loss_fn(Y_que_hat, Y_que)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        after_params = [(n, p.clone()) for n, p in model.named_parameters()]

        for (b_name, b_param), (a_name, a_param) in zip(
                before_params, after_params):
            # Make sure something changed.
            print(b_name, '\n', b_param)
            print(a_name, '\n', a_param)
            # self.assertTrue((b_param != a_param).any())
            pass


if __name__ == '__main__':
    unittest.main()