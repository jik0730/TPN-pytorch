# Base code is from https://github.com/cs230-stanford/cs230-code-examples
import logging

import numpy as np
from torch.autograd import Variable
from src.data_loader import fetch_dataloaders


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
                        'test' if evaluate on 'meta-testing' TODO 'meta-validating'
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

        # convert to torch Variables TODO do we need this?
        X_sup, Y_sup = Variable(X_sup), Variable(Y_sup)
        X_que, Y_que = Variable(X_que), Variable(Y_que)

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
    # TODO Evaluate trained model.
    pass