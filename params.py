import datetime
import os
from collections import defaultdict

import torch


###############################################
# EXP_NAME = "Simple-HWTF-Better-Loss-3-layer"
# EXP_NAME = "Simple-HWTF-Better-Loss-Small"
EXP_NAME = "LSTM"
# TODO: When you change the model parameters, change the EXP_NAME!

# training parameters
BATCH_SIZE = 1
LR = 0.0005
EPOCHS = 10000
RESUME = True

# transformer parameters
TF_D_MODEL = 64
TF_DROPOUT = 0
TF_N_HEADS = 4
TF_DIM_FEEDFORWARD = 64
TF_ENC_LAYERS = 1
TF_DEC_LAYERS = 1

# convolution parameter
KERNEL_SIZE = 10
STRIDE = 5
OUT_CHANNELS=8

ADD_NOISE = False

# mixture parameter
MX_N_GAUSSIANS=10


GRADIENT_THRESHOLD = 0.5

# data and model path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_MODEL = 10
SAVE_MODEL_HISTORY = 1000

PROCESED_DATA_DIR = "data/processed"

MODELS_DIR = os.path.join('saved_models', EXP_NAME)
model_architecture_path = os.path.join(MODELS_DIR, 'model.pth')
model_state_dict_path = os.path.join(MODELS_DIR, 'model_state_dict.pth')
def model_state_dict_epoch_path(epoch):
    return os.path.join(MODELS_DIR, 'model_state_dict_{}.pth'.format(epoch))

# tsrokes and characters
ALPHABET = [
    '\x00', ' ', '!', '"', '#', "'", '(', ')', ',', '-', '.',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';',
    '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z'
]
ALPHABET_SIZE = len(ALPHABET)
alphabet_ord = list(map(ord, ALPHABET))
alpha_to_num = defaultdict(int, list(map(reversed, enumerate(ALPHABET))))
num_to_alpha = dict(enumerate(alphabet_ord))

D_STROKE = 3

MAX_STROKE_LEN = 1200
MAX_CHAR_LEN = 75
