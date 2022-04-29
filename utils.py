import torch
import matplotlib.pyplot as plt
from params import *
import numpy as np

class Debugger:
    def __init__(self, function_name, debug=True):
        self.function_name = function_name
        self.debug = debug

    def print(self, *argv):
        if self.debug:
            print("[{}]".format(self.function_name), argv, sep=" ")


def create_pad_mask_by_len(max_seq_len, tensor_len) -> torch.tensor:
    """_summary_

    Args:
        tensor (N, L, ...):
        tensor_len (N,):

    Returns:
        torch.tensor: the pad_mask
    """

    L = max_seq_len
    N = tensor_len.shape[0]
    pad_mask = torch.full([N, L], False)
    for i, l in enumerate(tensor_len):
        if l >= 0:
            pad_mask[i, l:] = True
        else:
            pad_mask[i, :] = True

    return pad_mask


def create_pad_mask(tensor, tensor_len) -> torch.tensor:
    """_summary_

    Args:
        tensor (N, L, ...):
        tensor_len (N,):

    Returns:
        torch.tensor: the pad_mask
    """

    N, L = tensor.shape[0], tensor.shape[1]
    pad_mask = torch.full([N, L], False)
    for i, l in enumerate(tensor_len):
        if l >= 0:
            pad_mask[i, l:] = True
        else:
            pad_mask[i, :] = True

    return pad_mask

def draw(output, filename=None):
    output = output.cpu().detach()

    x = output.squeeze()
    abs_x = torch.cumsum(x, dim=0)

    coord_x = abs_x[:, 0]
    coord_y = abs_x[:, 1]
    plt.scatter(coord_x.cpu(), coord_y.cpu(), 3)
    plt.axis('equal')
    if filename != None:
        plt.savefig(filename)
    plt.show()
    plt.close()

def predict_no_guidance(model, batch):
    """_summary_

    Args:
        batch_seq_char (N, L): sequence of character using the ALPHABET indexing
        batch_seq_char_len (N): lenght of each sequence in the batch
    """

    tgt = torch.zeros(size=[1, 1, 2]).to(DEVICE)
    sequence_length = batch['c_len'].item() * 25

    with torch.no_grad():
        while tgt.shape[1] < sequence_length:
            batch['x'] = tgt
            batch['x_len'] = np.asarray([tgt.shape[1]])
            out = model(batch)
            tgt = torch.cat([tgt, out[:, [-1], :]], 1)

    return tgt

def num_to_string(c):
    "".join([str(num_to_alpha[n]) for n in c])
