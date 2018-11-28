# Base code is from https://github.com/cs230-stanford/cs230-code-examples
import argparse
import os
import logging

import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from tqdm import tqdm
from src.model import TPN
from src.model import metrics
from src.data_loader import split_omniglot_characters
from src.data_loader import load_imagenet_images
from src.data_loader import OmniglotTask
from src.data_loader import ImageNetTask
from src.data_loader import fetch_dataloaders
from evaluate import evaluate

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
    default=None,
    help="Optional, name of the file in --model_dir containing weights to \
          reload before training")  # 'best' or 'train'


def train_single_task(model, optimizer, loss_fn, dataloaders, metrics, params):
    """
    Train the model on a single few-shot task.
    
    Args:
        model: TPN model
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a loss function
        dataloaders: (dict) a dict of DataLoader objects that fetches both of 
                     support set and query set
        metrics: (dict) a dictionary of functions that compute a metric using 
                 the output and labels of each batch
        params: (Params) hyperparameters
    """
    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # support set and query set for a single few-shot task
    dl_sup = dataloaders['train']
    dl_que = dataloaders['test']
    X_sup, Y_sup = dl_sup.__iter__().next()
    X_que, Y_que = dl_que.__iter__().next()

    # convert to torch Variables TODO do we need this? maybe net
    # X_sup, Y_sup = Variable(X_sup), Variable(Y_sup)
    # X_que, Y_que = Variable(X_que), Variable(Y_que)

    # move to GPU if available
    if params.cuda:
        X_sup, Y_sup = X_sup.cuda(async=True), Y_sup.cuda(async=True)
        X_que, Y_que = X_que.cuda(async=True), Y_que.cuda(async=True)

    # compute model output and loss
    Y_que_hat = model(X_sup, Y_sup, X_que)
    loss = loss_fn(Y_que_hat, Y_que)

    # clear previous gradients, compute gradients of all variables wrt loss
    optimizer.zero_grad()
    loss.backward()

    # performs updates using calculated gradients
    optimizer.step()

    # NOTE debugging
    # for name, param in model.named_parameters():
    #     print(name, '\n', param.grad)

    return loss.item()


def train_and_evaluate(model,
                       meta_train_classes,
                       meta_val_classes,
                       meta_test_classes,
                       task_type,
                       optimizer,
                       loss_fn,
                       metrics,
                       params,
                       model_dir,
                       restore_file=None):
    """
    Train the model and evaluate every `save_summary_steps`.

    Args:
        model: TPN model
        meta_train_classes: (list) the classes for meta-training
        meta_val_classes: (list) the classes for meta-validating
        meta_test_classes: (list) the classes for meta-testing
        task_type: (subclass of FewShotTask) a type for generating tasks
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a loss function
        metrics: (dict) a dictionary of functions that compute a metric using 
                 the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from
                      (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir,
                                    args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    # params information
    num_classes = params.num_classes
    num_samples = params.num_samples
    num_query = params.num_query

    # validation accuracy
    best_val_loss = float('inf')

    # For plotting to see summerized training procedure
    plot_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    with tqdm(total=params.num_episodes) as t:
        for episode in range(params.num_episodes):
            # Run one episode
            logging.info("Episode {}/{}".format(episode + 1,
                                                params.num_episodes))

            # Train a model on a single task (episode).
            # TODO meta-batch of tasks
            task = task_type(meta_train_classes, num_classes, num_samples,
                             num_query)
            dataloaders = fetch_dataloaders(['train', 'test'], task)

            _ = train_single_task(model, optimizer, loss_fn, dataloaders,
                                  metrics, params)
            # print(episode, _)

            # TODO Evaluate for validation
            # Evaluate on train and test dataset given a number of tasks (params.num_steps)
            if (episode + 1) % params.save_summary_steps == 0:
                train_metrics = evaluate(model, loss_fn, meta_train_classes,
                                         task_type, metrics, params, 'train')
                val_metrics = evaluate(model, loss_fn, meta_val_classes,
                                       task_type, metrics, params, 'val')
                test_metrics = evaluate(model, loss_fn, meta_test_classes,
                                        task_type, metrics, params, 'test')

                train_loss = train_metrics['loss']
                val_loss = val_metrics['loss']
                test_loss = test_metrics['loss']
                train_acc = train_metrics['accuracy']
                val_acc = val_metrics['accuracy']
                test_acc = test_metrics['accuracy']

                is_best = val_loss <= best_val_loss

                # Save weights
                utils.save_checkpoint({
                    'episode': episode + 1,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict()
                },
                                      is_best=is_best,
                                      checkpoint=model_dir)

                # If best_test, best_save_path
                if is_best:
                    logging.info("- Found new best accuracy")
                    best_val_loss = val_loss

                    # Save best test metrics in a json file in the model directory
                    best_train_json_path = os.path.join(
                        model_dir, "metrics_train_best_weights.json")
                    utils.save_dict_to_json(train_metrics,
                                            best_train_json_path)
                    best_val_json_path = os.path.join(
                        model_dir, "metrics_val_best_weights.json")
                    utils.save_dict_to_json(val_metrics, best_val_json_path)
                    best_test_json_path = os.path.join(
                        model_dir, "metrics_test_best_weights.json")
                    utils.save_dict_to_json(test_metrics, best_test_json_path)

                # Save latest test metrics in a json file in the model directory
                last_train_json_path = os.path.join(
                    model_dir, "metrics_train_last_weights.json")
                utils.save_dict_to_json(train_metrics, last_train_json_path)
                last_val_json_path = os.path.join(
                    model_dir, "metrics_val_last_weights.json")
                utils.save_dict_to_json(val_metrics, last_val_json_path)
                last_test_json_path = os.path.join(
                    model_dir, "metrics_test_last_weights.json")
                utils.save_dict_to_json(test_metrics, last_test_json_path)

                plot_history['train_loss'].append(train_loss)
                plot_history['train_acc'].append(train_acc)
                plot_history['val_loss'].append(val_loss)
                plot_history['val_acc'].append(val_acc)
                plot_history['test_loss'].append(test_loss)
                plot_history['test_acc'].append(test_acc)
                utils.plot_training_results(args.model_dir, plot_history)

                t.set_postfix(
                    tr_acc='{:05.3f}'.format(train_acc),
                    te_acc='{:05.3f}'.format(test_acc),
                    tr_loss='{:05.3f}'.format(train_loss),
                    te_loss='{:05.3f}'.format(test_loss))
                print('\n')

            t.update()


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    SEED = params.SEED
    alpha = params.alpha
    lr = params.learning_rate
    num_episodes = params.num_episodes

    # Use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(SEED)
    if params.cuda: torch.cuda.manual_seed(SEED)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # NOTE These params are only applicable to pre-specified model architecture.
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # fetch loss function and metrics
    loss_fn = nn.NLLLoss()
    model_metrics = metrics

    # Train the model
    logging.info("Starting training for {} episode(s)".format(num_episodes))
    train_and_evaluate(model, meta_train_classes, meta_val_classes,
                       meta_test_classes, task_type, optimizer, loss_fn,
                       model_metrics, params, args.model_dir,
                       args.restore_file)
