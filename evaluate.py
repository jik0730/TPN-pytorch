# Base code is from https://github.com/cs230-stanford/cs230-code-examples
import logging
import os
import argparse

import torch
import torch.nn as nn
import numpy as np
import utils
from src.model import TPN
from src.model import metrics
from src.data_loader import split_omniglot_characters
from src.data_loader import load_imagenet_images
from src.data_loader import OmniglotTask
from src.data_loader import ImageNetTask
from src.data_loader import fetch_dataloaders

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    default='data/Omniglot',
    help="Directory containing the dataset")
parser.add_argument(
    '--model_dir',
    default='experiments/base_model',
    help="Directory containing params.json")
parser.add_argument(
    '--restore_file',
    default='best',
    help="name of the file in --model_dir containing weights to load")


def evaluate(model, loss_fn, meta_classes, task_type, metrics, params, split):
    """
    Evaluate the model on `num_steps` batches.
    
    Args:
        model: TPN model
        loss_fn: a loss function
        meta_classes: (list) a list of classes to be evaluated in meta-training or meta-testing
        task_type: (subclass of FewShotTask) a type for generating tasks
        metrics: (dict) a dictionary of functions that compute a metric using 
                 the output and labels of each batch
        params: (Params) hyperparameters
        split: (string) 'train' if evaluate on 'meta-training' and 
                        'val' if evaluate on 'meta-validating' and
                        'test' if evaluate on 'meta-testing'
    """
    # params information
    SEED = params.SEED
    num_classes = params.num_classes
    num_samples = params.num_samples
    num_query = params.num_query
    num_steps = params.num_steps

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for episode in range(num_steps):
        # Make a single task
        # Make dataloaders to load support set and query set
        task = task_type(meta_classes, num_classes, num_samples, num_query)
        dataloaders = fetch_dataloaders(['train', 'test'], task)
        dl_sup = dataloaders['train']
        dl_que = dataloaders['test']
        X_sup, Y_sup = dl_sup.__iter__().next()
        X_que, Y_que = dl_que.__iter__().next()

        # move to GPU if available
        if params.cuda:
            X_sup, Y_sup = X_sup.cuda(async=True), Y_sup.cuda(async=True)
            X_que, Y_que = X_que.cuda(async=True), Y_que.cuda(async=True)

        # Evaluate the model given a task
        Y_que_hat = model(X_sup, Y_sup, X_que)
        loss = loss_fn(Y_que_hat, Y_que)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        Y_que_hat = Y_que_hat.data.cpu().numpy()
        Y_que = Y_que.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {
            metric: metrics[metric](Y_que_hat, Y_que)
            for metric in metrics
        }
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {
        metric: np.mean([x[metric] for x in summ])
        for metric in summ[0]
    }
    metrics_string = " ; ".join(
        "{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- [" + split.upper() + "] Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the trained model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    SEED = params.SEED

    # use GPU if available
    params.cuda = torch.cuda.is_available()  # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(SEED)
    if params.cuda: torch.cuda.manual_seed(SEED)

    # Split meta-training and meta-testing characters
    if 'Omniglot' in args.data_dir and params.dataset == 'Omniglot':
        params.in_channels = 1
        params.in_features_fc = 1
        (meta_train_classes, meta_val_classes,
         meta_test_classes) = split_omniglot_characters(args.data_dir, SEED)
        task_type = OmniglotTask
    elif ('miniImageNet' in args.data_dir or
          'tieredImageNet' in args.data_dir) and params.dataset == 'ImageNet':
        params.in_channels = 3
        params.in_features_fc = 4
        (meta_train_classes, meta_val_classes,
         meta_test_classes) = load_imagenet_images(args.data_dir)
        task_type = ImageNetTask
    else:
        raise ValueError("I don't know your dataset")

    # Define the model and optimizer
    if params.cuda:
        model = TPN(params).cuda()
    else:
        model = TPN(params)

    # fetch loss function and metrics
    loss_fn = nn.NLLLoss()
    model_metrics = metrics

    # Reload weights from the saved file
    utils.load_checkpoint(
        os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, meta_test_classes, task_type,
                            metrics, params, 'test')
    print(test_metrics)