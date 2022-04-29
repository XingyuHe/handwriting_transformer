from __future__ import print_function
import os
import collections

import numpy as np
import torch

from sklearn.model_selection import train_test_split

from data.data_frame import DataFrame
from params import *

class DataReader(object):

    def __init__(self, data_dir, batch_first=True):
        data_cols = ['x', 'x_len', 'c', 'c_len', 'w_id']
        data = []

        for col in data_cols:
            data.append(
                torch.from_numpy(
                    np.load(os.path.join(data_dir, '{}.npy'.format(col)))
                ).to(DEVICE)
            )

        data[2] = data[2].to(torch.int64) # int8 is not compatible with Pytorch

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
            batch['x_len'] = batch['x_len'] - 1
            max_x_len = torch.max(batch['x_len'])
            max_c_len = torch.max(batch['c_len'])
            batch['y'] = batch['x'][:, 1:max_x_len + 1, :D_STROKE]
            batch['x'] = batch['x'][:, :max_x_len, :D_STROKE]
            batch['c'] = batch['c'][:, :max_c_len]
            yield batch

class WriterDataReader(object):

    def __init__(self, data_dir, wid_idx):
        data_cols = ['x', 'x_len', 'c', 'c_len', 'w_id']
        data = []

        for col in data_cols:
            data.append(
                torch.from_numpy(
                    np.load(os.path.join(data_dir, '{}.npy'.format(col)))
                ).to(DEVICE)
            )

        data[2] = data[2].to(torch.int64) # int8 is not compatible with Pytorch

        self.widToIdx = collections.defaultdict(list)
        for idx, wid in enumerate(data[-1].tolist()):
            self.widToIdx[wid].append(idx)

        self.index = torch.tensor(self.widToIdx[data[-1].tolist()[wid_idx]]).to(DEVICE)
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
            batch['x_len'] = batch['x_len'] - 1
            max_x_len = torch.max(batch['x_len'])
            max_c_len = torch.max(batch['c_len'])
            batch['y'] = batch['x'][:, 1:max_x_len + 1, :D_STROKE]
            batch['x'] = batch['x'][:, :max_x_len, :D_STROKE]
            batch['c'] = batch['c'][:, :max_c_len]
            yield batch



class DataReaderBySeqLen:

    def __init__(self, data_dir) -> None:
        data_cols = ['x', 'x_len', 'c', 'c_len', 'w_id']
        data = []

        for col in data_cols:
            data.append(
                torch.from_numpy(
                    np.load(os.path.join(data_dir, '{}.npy'.format(col)))
                ).to(DEVICE)
            )

        data[2] = data[2].to(torch.int64) # int8 is not compatible with Pytorch

        data_cols.append('y')
        data.append(data[0][:, 1:, :])
        data[0] = data[0][:, :-1, :]
        data[1] -= 1

        self.train_data, self.test_data = self.train_test_split(data=data, train_size=0.95, random_state=2018)
        self.data_cols = data_cols

    def get_size_to_idx(self, data):

        sizes_to_idx = defaultdict(list)
        for idx, d in enumerate(data):
            sizes_to_idx[d.shape[0]].append(idx)

        return sizes_to_idx

    def batch_generator(self, data, sizes_to_idx):


        for size, idxes in sizes_to_idx.items():
            batch = {}
            batch["x"] = torch.index_select(data[0], 0, torch.tensor(idxes).to(DEVICE))[:, :size, :]
            batch["x_len"] = torch.index_select(data[1], 0, torch.tensor(idxes).to(DEVICE))[:, :size, :]
            batch["c"] = torch.index_select(data[2], 0, torch.tensor(idxes).to(DEVICE))
            batch["c_len"] = torch.index_select(data[3], 0, torch.tensor(idxes).to(DEVICE))
            batch["y"] = torch.index_select(data[4], 0, torch.tensor(idxes).to(DEVICE))

            yield batch



    def train_test_split(self, data, train_size, random_state=np.random.randint(1000), stratify=None):
        train_idx, test_idx = train_test_split(
            np.arange(data[0].shape[0]),
            train_size=train_size,
            random_state=random_state,
            stratify=stratify
        )
        train_data = [mat[train_idx] for mat in data]
        test_data = [mat[test_idx] for mat in data]
        return train_data, test_data

    def train_batch_generator(self):
        return self.batch_generator(self.train_data, self.get_size_to_idx(self.train_data))

    def val_batch_generator(self, batch_size):
        return self.batch_generator(self.val_data, self.get_size_to_idx(self.val_data))

    def test_batch_generator(self, batch_size):
        return self.batch_generator(self.test_data, self.get_size_to_idx(self.test_data))


