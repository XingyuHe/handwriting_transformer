from calendar import EPOCH
import os
from tarfile import ENCODING
import time
from tkinter import ANCHOR

from data.data import *
from models.model import *

import torch

from params import *
from utils import *


os.environ["WANDB_API_KEY"] = "5d251ce3efa3311b8bcbea9d51c3c54e9b4b4ac5"

import wandb

def train_loop(model, optimizer, batch):
    optimizer.zero_grad()
    es, pis, mu1s, mu2s, sigma1s, sigma2s, rhos  = model(batch)
    Pr = model.gaussianMixture(batch['y'], pis, mu1s, mu2s, sigma1s, sigma2s, rhos)
    mse_loss = model.loss_fn(Pr, es)
    loss = mse_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_THRESHOLD)
    optimizer.step()
    return loss

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

    print("Experiment name: ", EXP_NAME)

    print("========================== params.py ==========================")
    with open('params.py', 'r') as f:
        print(f.read())
    print("===============================================================")

    if os.path.isdir(MODELS_DIR) and RESUME:
        model = torch.load(model_architecture_path)
        if os.path.isfile(model_state_dict_path):
            model.load_state_dict(torch.load(model_state_dict_path))
        print (MODELS_DIR+' : Model loaded Successfully')
    else:
        if not os.path.isdir(MODELS_DIR):
            os.mkdir(MODELS_DIR)
        model = LSTM_HW()
        torch.save(model, model_architecture_path)

    model.to(DEVICE)
    if next(model.parameters()).is_cuda == False:
        raise ValueError("model is not on cuda GPU")

    # set up data pipeline
    # data_reader = DataReader(PROCESED_DATA_DIR)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR, betas=(0.0, 0.999), weight_decay=0, eps=1e-8
    )


    data_reader = DataReader("data/processed")

    # training loop
    for epoch in range(EPOCHS + 1):

        train_data = data_reader.train_batch_generator(BATCH_SIZE)
        val_data = data_reader.test_batch_generator(BATCH_SIZE)
        start_time = time.time()
        sample_cnt = 0

        wandb_log = {}
        wandb_log["train_loss"] = 0


        for batch in train_data:
            loss = train_loop(model, optimizer, batch)

            sample_cnt += BATCH_SIZE
            wandb_log["train_loss"] += batch['x'].shape[0] * loss.item()

        with torch.no_grad():
            wandb_log["val_loss"] = 0
            for batch in val_data:

                with torch.no_grad():
                    es, pis, mu1s, mu2s, sigma1s, sigma2s, rhos  = model(batch)
                    Pr = model.gaussianMixture(batch['y'], pis, mu1s, mu2s, sigma1s, sigma2s, rhos)
                    mse_loss = model.loss_fn(Pr, es)
                    wandb_log["val_loss"] += loss.item()

        wandb_log["train_loss"] = wandb_log["train_loss"]/data_reader.get_train_len()
        wandb_log["val_loss"] = wandb_log["val_loss"]/data_reader.get_val_len()

        wandb_log["epoch"] = epoch
        wandb_log['timeperepoch'] = time.time() - start_time

        print (wandb_log)
        wandb.log(wandb_log)

        if epoch % SAVE_MODEL == 0:
            test_words = "hello"
            test_batch = {"c": torch.tensor([alpha_to_num[c] for c in test_words]).unsqueeze(0).to(DEVICE), "c_len": torch.tensor([len(test_words)]).to(DEVICE)}


            torch.save(model.state_dict(), model_state_dict_path)
            out = model.generate_sequence()
            img_path = os.path.join(MODELS_DIR, "test_img_epoch-{}.jpg".format(epoch))
            draw(out, img_path)
            wandb.log({"test_img_epoch-{}".format(epoch): wandb.Image(img_path)})

        if epoch % SAVE_MODEL_HISTORY == 0: torch.save(model.state_dict(), model_state_dict_epoch_path(epoch))


if __name__ == "__main__":
    main()
