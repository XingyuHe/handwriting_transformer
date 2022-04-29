from data.data import *
dr = DataReaderBySeqLen("data/processed")
for b in dr.train_batch_generator():
    print(b['x'].shape, b['y'].shape, b['c'].shape, b['x_len'].shape, b['c_len'].shape)
