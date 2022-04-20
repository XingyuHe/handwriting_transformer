# %%
from pathlib import Path
import os

import pandas as pd
import numpy as np

import torch
from data import BaseDataReader
# %%
dr = BaseDataReader("processed")
# %%
for i in dr.train_batch_generator(1):
    print(i['x'].shape)
    print(i['y'].shape)
    print(i['c'].shape)
    print()

# %%
