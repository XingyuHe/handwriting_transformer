from __future__ import print_function
import os
import collections

import numpy as np
import torch

from data.data_frame import DataFrame
from params import *


class WriterDataReader(object):

    def __init__(self, data_dir):
        data_cols = ['x', 'x_len', 'c', 'c_len', 'w_id']
        data = []

        for col in data_cols:
            data.append(
                torch.from_numpy(
                    np.load(os.path.join(data_dir, '{}.npy'.format(col)))
                ).to(DEVICE)
            )

        data[2] = data[2].type(torch.int) # int8 is not compatible with Pytorch

        self.widToIdx = collections.defaultdict(list)
        for idx, wid in enumerate(data[-1].tolist()):
            self.widToIdx[wid].append(idx)

        self.index = torch.tensor(self.widToIdx[wid]).to(DEVICE)
        data = [
            torch.index_select(data[0], 0, self.index),
            torch.index_select(data[1], 0, self.index),
            torch.index_select(data[2], 0, self.index),
            torch.index_select(data[3], 0, self.index),
            torch.index_select(data[4], 0, self.index)
        ]

        self.test_df = DataFrame(columns=data_cols, data=data)
        self.train_df, self.val_df = self.test_df.train_test_split(train_size=0.95, random_state=2018)

        print('train size', len(self.train_df))
        print('val size', len(self.val_df))
        print('test size', len(self.test_df))

    def get_train_len(self):
        return len(self.train_df)

    def get_val_len(self):
        return len(self.val_df)

    def get_test_len(self):
        return len(self.test_df)

    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=True,
            num_epochs=1,
            mode='train'
        )

    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=True,
            num_epochs=1,
            mode='val'
        )

    def test_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.test_df,
            shuffle=False,
            num_epochs=1,
            mode='test'
        )

    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=10000, mode='train'):
        gen = df.batch_generator(
            batch_size=batch_size,
            shuffle=shuffle,
            num_epochs=num_epochs,
            allow_smaller_final_batch=(mode == 'test')
        )
        for batch in gen:
            # batch['x_len'] = batch['x_len'] - 1
            max_x_len = torch.max(batch['x_len'])
            max_c_len = torch.max(batch['c_len'])
            # batch['y'] = batch['x'][:, 1:max_x_len + 1, :]
            batch['x'] = batch['x'][:, :max_x_len, :]
            batch['c'] = batch['c'][:, :max_c_len]
            yield batch

class DataReader(object):

    def __init__(self, data_dir):
        data_cols = ['x', 'x_len', 'c', 'c_len', 'w_id']
        data = []

        for col in data_cols:
            data.append(
                torch.from_numpy(
                    np.load(os.path.join(data_dir, '{}.npy'.format(col)))
                ).to(DEVICE)
            )

        data[2] = data[2].type(torch.int) # int8 is not compatible with Pytorch

        self.test_df = DataFrame(columns=data_cols, data=data)
        self.train_df, self.val_df = self.test_df.train_test_split(train_size=0.95, random_state=2018)

        print('train size', len(self.train_df))
        print('val size', len(self.val_df))
        print('test size', len(self.test_df))

    def get_train_len(self):
        return len(self.train_df)

    def get_val_len(self):
        return len(self.val_df)

    def get_test_len(self):
        return len(self.test_df)

    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=True,
            num_epochs=1,
            mode='train'
        )

    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=True,
            num_epochs=1,
            mode='val'
        )

    def test_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.test_df,
            shuffle=False,
            num_epochs=1,
            mode='test'
        )

    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=10000, mode='train'):
        gen = df.batch_generator(
            batch_size=batch_size,
            shuffle=shuffle,
            num_epochs=num_epochs,
            allow_smaller_final_batch=(mode == 'test')
        )
        for batch in gen:
            # batch['x_len'] = batch['x_len'] - 1
            max_x_len = torch.max(batch['x_len'])
            max_c_len = torch.max(batch['c_len'])
            # batch['y'] = batch['x'][:, 1:max_x_len + 1, :]
            batch['x'] = batch['x'][:, :max_x_len, :]
            batch['c'] = batch['c'][:, :max_c_len]
            yield batch