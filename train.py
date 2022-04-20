from calendar import EPOCH
import os
from tarfile import ENCODING
import time
from tkinter import ANCHOR

from data.data import *
from models.model import TFHW

import torch

from params import *


os.environ["WANDB_API_KEY"] = "5d251ce3efa3311b8bcbea9d51c3c54e9b4b4ac5"

import wandb

def main():
    wandb.init(project="handwriting-transformers", entity="xh2513")
    wandb.config = {
        "LR": LR,
        "EPOCHS": EPOCHS,
        "BATCH_SIZE":BATCH_SIZE,
        "TF_D_MODEL" : TF_D_MODEL,
        "TF_DROPOUT" : TF_DROPOUT,
        "TF_N_HEADS" : TF_N_HEADS,
        "TF_DIM_FEEDFORWARD" : TF_DIM_FEEDFORWARD,
        "TF_ENC_LAYERS" : TF_ENC_LAYERS,
        "TF_DEC_LAYERS" : TF_DEC_LAYERS,
        "ADD_NOISE" : ADD_NOISE,
    }

    # set up model directory && instantiate the model
    model = None

    if os.path.isdir(MODELS_DIR) and RESUME:
        model = torch.load(model_architecture_path)
        if os.path.isfile(model_state_dict_path):
            model.load_state_dict(torch.load(model_state_dict_path))
        print (MODELS_DIR+' : Model loaded Successfully')
    else:
        if not os.path.isdir(MODELS_DIR):
            os.mkdir(MODELS_DIR)
        model = TFHW()
        torch.save(model, model_architecture_path)

    model.to(DEVICE)
    if next(model.parameters()).is_cuda == False:
        raise ValueError("model is not on cuda GPU")

    # set up data pipeline
    data_reader = DataReader(PROCESED_DATA_DIR)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR, betas=(0.0, 0.999), weight_decay=0, eps=1e-8
    )


    # training loop
    for epoch in range(EPOCHS):

        start_time = time.time()
        train_data = data_reader.train_batch_generator(batch_size=BATCH_SIZE)

        sample_cnt = 0

        wandb_log = {}
        wandb_log["train_loss"] = 0
        wandb_log["train_mse_loss"] = 0
        wandb_log["train_logit_loss"] = 0


        for batch in train_data:
            print(batch["x"].shape)
            optimizer.zero_grad()
            output = model(batch)
            mse_loss, logit_loss = model.loss_fn(output, batch['x'])
            loss = mse_loss + logit_loss
            loss.backward()
            optimizer.step()

            sample_cnt += BATCH_SIZE
            print ('\t', {'CNT': sample_cnt,'LOSS': loss})
            wandb_log["train_loss"] += batch['x'].shape[0] * loss.item()
            wandb_log["train_mse_loss"] += batch['x'].shape[0] * mse_loss.item()
            wandb_log["train_logit_loss"] += batch['x'].shape[0] * logit_loss.item()
            # single batch fitting

        with torch.no_grad():
            wandb_log["val_loss"] = 0
            wandb_log["val_mse_loss"] = 0
            wandb_log["val_logit_loss"] = 0
            for batch in data_reader.val_batch_generator(batch_size=1):
                mse_loss, logit_loss = model.loss_fn(model(batch), batch['x'])
                loss = mse_loss + logit_loss
                wandb_log["val_loss"] += loss.item()
                wandb_log["val_mse_loss"] += mse_loss.item()
                wandb_log["val_logit_loss"] += logit_loss.item()

        wandb_log["train_loss"] = wandb_log["train_loss"]/data_reader.get_train_len()
        wandb_log["train_mse_loss"] = wandb_log["train_mse_loss"]/data_reader.get_train_len()
        wandb_log["train_logit_loss"] = wandb_log["train_logit_loss"]/data_reader.get_train_len()

        wandb_log["val_loss"] = wandb_log["val_loss"]/data_reader.get_val_len()
        wandb_log["val_mse_loss"] = wandb_log["val_mse_loss"]/data_reader.get_val_len()
        wandb_log["val_logit_loss"] = wandb_log["val_logit_loss"]/data_reader.get_val_len()

        wandb_log["epoch"] = epoch
        wandb_log['timeperepoch'] = time.time() - start_time

        wandb.log(wandb_log)

        print (wandb_log)

        if epoch % SAVE_MODEL == 0: torch.save(model.state_dict(), model_state_dict_path)
        if epoch % SAVE_MODEL_HISTORY == 0: torch.save(model.state_dict(), model_state_dict_epoch_path(epoch))


if __name__ == "__main__":
    main()