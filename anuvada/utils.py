"""
Code credits to
https://github.com/henryre/pytorch-fitmodule
"""

import numpy as np
import sys
import torch

from functools import partial


##### Data utils #####

def make_batches(size, batch_size):
    """github.com/fchollet/keras/blob/master/keras/engine/training.py"""
    num_batches = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, num_batches)]


##### Logging #####

def add_metrics_to_log(log, metrics, y_true, y_pred, prefix=''):
    for metric in metrics:
        q = metric(y_true, y_pred)
        log[prefix + metric.__name__] = q
    return log


def log_to_message(log, precision=4):
    fmt = "{0}: {1:." + str(precision) + "f}"
    return "    ".join(fmt.format(k, v) for k, v in log.items())


class ProgressBar(object):
    """Cheers @ajratner"""

    def __init__(self, N, length=40):
        # Protect against division by zero
        self.N      = max(1, N)
        self.nf     = float(self.N)
        self.length = length
        # Precalculate the i values that should trigger a write operation
        self.ticks = set([round(i/100.0 * N) for i in range(101)])
        self.ticks.add(N-1)
        self.bar(0)

    def bar(self, i, message=""):
        """Assumes i ranges through [0, N-1]"""
        if i in self.ticks:
            b = int(np.ceil(((i+1) / self.nf) * self.length))
            sys.stdout.write("\r[{0}{1}] {2}%\t{3}".format(
                "="*b, " "*(self.length-b), int(100*((i+1) / self.nf)), message
            ))
            sys.stdout.flush()

    def close(self, message=""):
        # Move the bar to 100% before closing
        self.bar(self.N-1)
        sys.stdout.write("{0}\n\n".format(message))
        sys.stdout.flush()


def save_model(model_object, filepath):
    torch.save(model_object.state_dict(), filepath)
    print 'Model saved.'
    return None


def load_model(model_object, filepath):
    weights = torch.load(filepath)
    model_object.load_state_dict(weights)
    print 'Model loaded.'
    return None

def to_multilabel(list_of_ids, n_classes):
    label_vector = np.zeros(n_classes,dtype=np.int)
    for idx in list_of_ids:
        label_vector[idx] = 1
    return label_vector

