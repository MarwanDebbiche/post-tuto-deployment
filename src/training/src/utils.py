import math
import json
import re
import numpy as np
from sklearn import metrics

# text-preprocessing


def lower(text):
    return text.lower()


def remove_hashtags(text):
    clean_text = re.sub(r'#[A-Za-z0-9_]+', "", text)
    return clean_text


def remove_user_mentions(text):
    clean_text = re.sub(r'@[A-Za-z0-9_]+', "", text)
    return clean_text


def remove_urls(text):
    clean_text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    return clean_text


preprocessing_setps = {
    'remove_hashtags': remove_hashtags,
    'remove_urls': remove_urls,
    'remove_user_mentions': remove_user_mentions,
    'lower': lower
}


def process_text(steps, text):
    if steps is not None:
        for step in steps:
            text = preprocessing_setps[step](text)
    return text

# metrics // model evaluations


def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'f1' in list_metrics:
        output['f1'] = metrics.f1_score(y_true, y_pred, average='weighted')

    return output


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# preprocess input for prediction


def preprocess_input(args):
    raw_text = args.text
    steps = args.steps
    for step in steps:
        raw_text = preprocessing_setps[step](raw_text)

    number_of_characters = args.number_of_characters + \
        len(args.extra_characters)
    identity_mat = np.identity(number_of_characters)
    vocabulary = list(args.alphabet) + list(args.extra_characters)
    max_length = args.max_length

    processed_output = np.array([identity_mat[vocabulary.index(i)] for i in list(
        raw_text[::-1]) if i in vocabulary], dtype=np.float32)
    if len(processed_output) > max_length:
        processed_output = processed_output[:max_length]
    elif 0 < len(processed_output) < max_length:
        processed_output = np.concatenate((processed_output, np.zeros(
            (max_length - len(processed_output), number_of_characters), dtype=np.float32)))
    elif len(processed_output) == 0:
        processed_output = np.zeros(
            (max_length, number_of_characters), dtype=np.float32)
    return processed_output


# cyclic learning rate scheduling

def cyclical_lr(stepsize, min_lr=1.7e-3, max_lr=1e-2):

    # Scaler: we can adapt this if we do not want the triangular CLR
    def scaler(x): return 1.

    # Lambda function to calculate the LR
    def lr_lambda(it): return min_lr + (max_lr -
                                        min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda
