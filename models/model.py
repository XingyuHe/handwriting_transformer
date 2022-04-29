import encodings
from quopri import decodestring
from random import sample
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from params import *
from utils import *

import math

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self):
        super(PositionalEncoding, self).__init__()
        max_len = max(MAX_CHAR_LEN, MAX_STROKE_LEN)
        self.dropout = nn.Dropout(p=TF_DROPOUT)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, TF_D_MODEL, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, TF_D_MODEL, 2) *
                             -(math.log(10000.0) / TF_D_MODEL))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)

class LSTM_HW(nn.Module):
    def __init__(self, hidden_size = 256, n_gaussians = 20, Kmixtures = 10, dropout = 0.2, alphabet_size = 64):
        super(LSTM_HW, self).__init__()

        self.Kmixtures = Kmixtures
        self.n_gaussians = n_gaussians
        self.alphabet_size = alphabet_size

        self.hidden_size1 = hidden_size
        self.hidden_size2 = hidden_size
        self.hidden_size3 = hidden_size

        # input_size1 includes x, y, eos and len(w_t_1) given by alphabet_size (see eq 52)
        self.input_size1 = 3 + alphabet_size

        # input_size2 includes x, y, eos, len(w_t) given by alphabet_size (see eq 47) and hidden_size1
        self.input_size2 = 3 + alphabet_size + self.hidden_size1

        # input_size3 includes x, y, eos, len(w_t) given by alphabet_size (see eq 47) and hidden_size2
        self.input_size3 = 3 + alphabet_size + self.hidden_size2

        # See eq 52-53 to understand the input_sizes
        self.lstm1 = nn.LSTMCell(input_size= self.input_size1 , hidden_size = self.hidden_size1)
        self.lstm2 = nn.LSTMCell(input_size= self.input_size2 , hidden_size = self.hidden_size2)
        self.lstm3 = nn.LSTMCell(input_size= self.input_size3 , hidden_size = self.hidden_size3)

        # Window layer takes hidden layer of LSTM1 as input and outputs 3 * Kmixtures vectors
        self.window_layer = nn.Linear(self.hidden_size1, 3 * Kmixtures)

        # For gaussian mixtures
        self.z_e = nn.Linear(hidden_size, 1)
        self.z_pi = nn.Linear(hidden_size, n_gaussians)
        self.z_mu1 = nn.Linear(hidden_size, n_gaussians)
        self.z_mu2 = nn.Linear(hidden_size, n_gaussians)
        self.z_sigma1 = nn.Linear(hidden_size, n_gaussians)
        self.z_sigma2 = nn.Linear(hidden_size, n_gaussians)
        self.z_rho = nn.Linear(hidden_size, n_gaussians)

        # Bias for sampling
        self.bias = 0

        # Saves hidden and cell states
        self.LSTMstates = None

    def gaussianMixture(self, y, pis, mu1s, mu2s, sigma1s, sigma2s, rhos):
        n_mixtures = pis.size(2)

        # Takes x1 and repeats it over the number of gaussian mixtures
        x1 = y[:,:, 0].repeat(n_mixtures, 1, 1).permute(1, 2, 0)
        # print("x1 shape ", x1.shape) # -> torch.Size([sequence_length, batch, n_gaussians])

        # first term of Z (eq 25)
        x1norm = ((x1 - mu1s) ** 2) / (sigma1s ** 2 )
        # print("x1norm shape ", x1.shape) # -> torch.Size([sequence_length, batch, n_gaussians])

        x2 = y[:,:, 1].repeat(n_mixtures, 1, 1).permute(1, 2, 0)
        # print("x2 shape ", x2.shape) # -> torch.Size([sequence_length, batch, n_gaussians])

        # second term of Z (eq 25)
        x2norm = ((x2 - mu2s) ** 2) / (sigma2s ** 2 )
        # print("x2norm shape ", x2.shape) # -> torch.Size([sequence_length, batch, n_gaussians])

        # third term of Z (eq 25)
        coxnorm = 2 * rhos * (x1 - mu1s) * (x2 - mu2s) / (sigma1s * sigma2s)

        # Computing Z (eq 25)
        Z = x1norm + x2norm - coxnorm

        # Gaussian bivariate (eq 24)
        N = torch.exp(-Z / (2 * (1 - rhos ** 2))) / (2 * np.pi * sigma1s * sigma2s * (1 - rhos ** 2) ** 0.5)
        # print("N shape ", N.shape) # -> torch.Size([sequence_length, batch, n_gaussians])

        # Pr is the result of eq 23 without the eos part
        Pr = pis * N
        # print("Pr shape ", Pr.shape) # -> torch.Size([sequence_length, batch, n_gaussians])
        Pr = torch.sum(Pr, dim=2)
        # print("Pr shape ", Pr.shape) # -> torch.Size([sequence_length, batch])

        if use_cuda:
            Pr = Pr.cuda()

        return Pr

    def loss_fn(self, Pr, y, es):
        loss1 = - torch.log(Pr + eps) # -> torch.Size([sequence_length, batch])
        bernouilli = torch.zeros_like(es) # -> torch.Size([sequence_length, batch])

        bernouilli = y[:, :, 2] * es + (1 - y[:, :, 2]) * (1 - es)

        loss2 = - torch.log(bernouilli + eps)
        loss = loss1 + loss2
        # print("loss shape", loss.shape) # -> torch.Size([sequence_length, batch])
        loss = torch.sum(loss, 0)
        # print("loss shape", loss.shape) # -> torch.Size([batch])

        return torch.mean(loss);



    def forward(self, batch, generate = False):
        x = batch['x'].transpose(0, 1)
        c = F.one_hot(batch['c'], num_classes=self.alphabet_size)
        c = c.to(torch.float)

        # sequence length
        sequence_length = x.shape[0]

        # number of batches
        n_batch = x.shape[1]

        # Soft window vector w at t-1
        w_t_1 = torch.ones(n_batch, self.alphabet_size) # torch.Size([n_batch, len(alphabet)])

        # Hidden and cell state for LSTM1
        h1_t = torch.zeros(n_batch, self.hidden_size1) # torch.Size([n_batch, hidden_size1])
        c1_t = torch.zeros(n_batch, self.hidden_size1) # torch.Size([n_batch, hidden_size1])

        # Kappa at t-1
        kappa_t_1 = torch.zeros(n_batch, self.Kmixtures) # torch.Size([n_batch, Kmixtures])

        # Hidden and cell state for LSTM2
        h2_t = torch.zeros(n_batch, self.hidden_size2) # torch.Size([n_batch, hidden_size2])
        c2_t = torch.zeros(n_batch, self.hidden_size2) # torch.Size([n_batch, hidden_size2])

        # Hidden and cell state for LSTM3
        h3_t = torch.zeros(n_batch, self.hidden_size3) # torch.Size([n_batch, hidden_size3])
        c3_t = torch.zeros(n_batch, self.hidden_size3) # torch.Size([n_batch, hidden_size3])

        if generate and self.LSTMstates != None:
            h1_t = self.LSTMstates["h1_t"]
            c1_t = self.LSTMstates["c1_t"]
            h2_t = self.LSTMstates["h2_t"]
            c2_t = self.LSTMstates["c2_t"]
            h3_t = self.LSTMstates["h3_t"]
            c3_t = self.LSTMstates["c3_t"]
            w_t_1 = self.LSTMstates["w_t_1"]
            kappa_t_1 = self.LSTMstates["kappa_t_1"]

        out = torch.zeros(sequence_length, n_batch, self.hidden_size3)

        # Phis and Ws allow to plot heatmaps of phi et w over time
        self.Phis = torch.zeros(sequence_length, c.shape[1])
        self.Ws = torch.zeros(sequence_length, self.alphabet_size)

        print("w_t_1", w_t_1.shape)
        w_t_1 = w_t_1.to(DEVICE)

        h1_t = h1_t.to(DEVICE)
        c1_t = c1_t.to(DEVICE)

        kappa_t_1 = kappa_t_1.to(DEVICE)

        h2_t = h2_t.to(DEVICE)
        c2_t = c2_t.to(DEVICE)

        h3_t = h3_t.to(DEVICE)
        c3_t = c3_t.to(DEVICE)

        out = out.to(DEVICE)

        for i in range(sequence_length):
            # ===== Computing 1st layer =====
            input_lstm1 = torch.cat((x[i], w_t_1), 1) # torch.Size([n_batch, input_size1])
            h1_t, c1_t = self.lstm1(input_lstm1, (h1_t, c1_t)) # torch.Size([n_batch, hidden_size1])

            # ===== Computing soft window =====
            window = self.window_layer(h1_t)

            # splits exp(window) into 3 tensors of torch.Size([n_batch, Kmixtures])
            # Eqs 48-51 of the paper
            alpha_t, beta_t, kappa_t = torch.chunk( torch.exp(window), 3, dim=1)
            kappa_t = 0.1 * kappa_t + kappa_t_1

            # updates kappa_t_1 for next iteration
            kappa_t_1 = kappa_t

            u = torch.arange(0,c.shape[1], out=kappa_t.new()).view(-1,1,1) # torch.Size([U_items, 1, 1])

            # Computing Phi(t, u)
            # Eq 46 of the paper
            # Keep in mind the (kappa_t - u).shape is torch.Size([U_items, n_batch, Kmixtures])
            # For example :
            ## (kappa_t - u)[0, 0, :] gives kappa_t[0, :]
            ## (kappa_t - u)[1, 0, :] gives kappa_t[0, :] - 1
            ## etc
            Phi = alpha_t * torch.exp(- beta_t * (kappa_t - u) ** 2) # torch.Size([U_items, n_batch, Kmixtures])
            Phi = torch.sum(Phi, dim = 2) # torch.Size([U_items, n_batch])
            Phi = torch.unsqueeze(Phi, 0) # torch.Size([1, U_items, n_batch])
            Phi = Phi.permute(2, 0, 1) # torch.Size([n_batch, 1, U_items])

            self.Phis[i, :] = Phi[0, 0, :] # To plot heatmaps

            # Computing wt
            # Eq 47 of the paper
            w_t = torch.matmul(Phi, c) # torch.Size([n_batch, 1, len(alphabet)])
            w_t = torch.squeeze(w_t, 1) # torch.Size([n_batch, len(alphabet)])

            self.Ws[i, :] = w_t[0, :] # To plot heatmaps

            # Update w_t_1 for next iteration
            w_t_1 = w_t

            # ===== Computing 2nd layer =====
            input_lstm2 = torch.cat((x[i], w_t, h1_t), 1) # torch.Size([n_batch, 3 + alphabet_size + hidden_size1])
            h2_t, c2_t = self.lstm2(input_lstm2, (h2_t, c2_t))


            # ===== Computing 3rd layer =====
            input_lstm3 = torch.cat((x[i], w_t, h2_t), 1) # torch.Size([n_batch, 3 + alphabet_size + hidden_size2])
            h3_t, c3_t = self.lstm3(input_lstm3, (h3_t, c3_t))
            out[i, :, :] = h3_t

        # ===== Computing MDN =====
        es = self.z_e(out)
        # print("es shape ", es.shape) # -> torch.Size([sequence_length, batch, 1])
        es = 1 / (1 + torch.exp(es))
        # print("es shape", es.shape) # -> torch.Size([sequence_length, batch, 1])

        pis = self.z_pi(out) * (1 + self.bias)
        # print("pis shape ", pis.shape) # -> torch.Size([sequence_length, batch, n_gaussians])
        pis = torch.softmax(pis, 2)
        # print(pis.shape) # -> torch.Size([sequence_length, batch, n_gaussians])

        mu1s = self.z_mu1(out)
        mu2s = self.z_mu2(out)
        # print("mu shape :  ", mu1s.shape) # -> torch.Size([sequence_length, batch, n_gaussians])

        sigma1s = self.z_sigma1(out)
        sigma2s = self.z_sigma2(out)
        # print("sigmas shape ", sigma1s.shape) # -> torch.Size([sequence_length, batch, n_gaussians])
        sigma1s = torch.exp(sigma1s - self.bias)
        sigma2s = torch.exp(sigma2s - self.bias)
        # print(sigma1s.shape) # -> torch.Size([sequence_length, batch, n_gaussians])

        rhos = self.z_rho(out)
        rhos = torch.tanh(rhos)
        # print("rhos shape ", rhos.shape) # -> torch.Size([sequence_length, batch, n_gaussians])

        es = es.squeeze(2)
        # print("es shape ", es.shape) # -> torch.Size([sequence_length, batch])

        # Hidden and cell states
        if generate:
            self.LSTMstates = {"h1_t": h1_t,
                              "c1_t": c1_t,
                              "h2_t": h2_t,
                              "c2_t": c2_t,
                              "h3_t": h3_t,
                              "c3_t": c3_t,
                              "w_t_1": w_t_1,
                              "kappa_t_1": kappa_t_1}

        return es, pis, mu1s, mu2s, sigma1s, sigma2s, rhos




    def generate_sample(self, mu1, mu2, sigma1, sigma2, rho):
        mean = [mu1, mu2]
        cov = [[sigma1 ** 2, rho * sigma1 * sigma2], [rho * sigma1 * sigma2, sigma2 ** 2]]

        x = np.float32(np.random.multivariate_normal(mean, cov, 1))
        return torch.from_numpy(x)


    def generate_sequence(self, c0, bias=10):
        x0 = torch.tensor([0, 0, 1]).unsqueeze(0).unsqueeze(0)
        sequence = x0
        sample = x0
        sequence_length = c0.shape[1] * 25

        print("Generating sequence ...")
        self.bias = bias

        for i in range(sequence_length):
            es, pis, mu1s, mu2s, sigma1s, sigma2s, rhos = self.forward(sample, c0, True)

            # Selecting a mixture
            pi_idx = np.random.choice(range(self.n_gaussians), p=pis[-1, 0, :].detach().cpu().numpy())

            # taking last parameters from sequence corresponding to chosen gaussian
            mu1 = mu1s[-1, :, pi_idx].item()
            mu2 = mu2s[-1, :, pi_idx].item()
            sigma1 = sigma1s[-1, :, pi_idx].item()
            sigma2 = sigma2s[-1, :, pi_idx].item()
            rho = rhos[-1, :, pi_idx].item()

            prediction = self.generate_sample(mu1, mu2, sigma1, sigma2, rho)
            eos = torch.distributions.bernoulli.Bernoulli(torch.tensor([es[-1, :].item()])).sample()

            sample = torch.zeros_like(x0) # torch.Size([1, 1, 3])
            sample[0, 0, 0] = prediction[0, 0]
            sample[0, 0, 1] = prediction[0, 1]
            sample[0, 0, 2] = eos

            sequence = torch.cat((sequence, sample), 0) # torch.Size([sequence_length, 1, 3])


        self.bias = 0
        self.LSTMstates = None

        return sequence.squeeze(1).detach().cpu().numpy()

class TFHW_Convolution(nn.Module):
    def __init__(self, d_model = TF_D_MODEL, d_stroke = D_STROKE, num_heads = TF_N_HEADS,
                    dim_feedforward=TF_DIM_FEEDFORWARD,
                 out_channels = OUT_CHANNELS, kernel_size = KERNEL_SIZE, stride = STRIDE, padding=KERNEL_SIZE,
                 alphabet_size = ALPHABET_SIZE, max_stroke_len = MAX_STROKE_LEN, debug=False) -> None:
        super(TFHW_Convolution, self).__init__()


        self.alphabet_size = alphabet_size
        self.max_stroke_len = max_stroke_len

        self.d_model = d_model

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.encoder = nn.Sequential(nn.Embedding(alphabet_size, d_model), PositionalEncoding())

        self.tgt_embed = nn.Sequential(nn.Linear(d_stroke, d_model), PositionalEncoding())
        self.tgt_conv_embed = nn.Linear(out_channels, d_model)
        self.conv1d = nn.Conv1d(
            in_channels=d_stroke, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding = padding
        )
        self.conv_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )
        self.conv_attn_mask = self.convolution_attention_mask().to(DEVICE)
        # shape: MAX_SEQ_LEN x MAX_SEQ_LEN
        self.multi_head_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = F.relu

        self.generator = nn.Linear(d_model, d_stroke)

        self._reset_parameters()
        self.debugger = Debugger("TFHW_Convolution", debug)

    def encode(self, c):
        encoding = self.encoder(c)
        self.debugger.print("encoding shape", encoding.shape)

        return encoding

    def decode(self, tgt, mem, conv_tgt_mask, mem_mask, conv_tgt_padding_mask, mem_padding_mask):

        self.debugger.print("tgt shape", tgt.shape)

        tgt_embed = self.tgt_embed(tgt)

        # convolve
        # convolution requires in B x Cin x Lin
        tgt_conv_in = tgt.transpose(-1, -2)

        self.debugger.print("tgt_conv_in shape", tgt_conv_in.shape)
        tgt_conv_out = self.conv1d(tgt_conv_in).transpose(-1, -2) # B x L out x C out
        tgt_conv_embed = self.tgt_conv_embed(tgt_conv_out)

        self.debugger.print("tgt_conv_embed shape", tgt_conv_embed.shape)

#         post_kernel_tgt_embed = tgt_embed[:, self.kernel_size:, :]
        self.debugger.print("tgt_embed shape", tgt_embed.shape)

        # tranformer block

        self.debugger.print("tgt_conv_embed ", tgt_conv_embed)
        self.debugger.print("tgt_embed ", tgt_embed)
        self.debugger.print("new conv_tgt_mask shape ", conv_tgt_mask[self.kernel_size:, :])
        self.debugger.print("new conv_tgt_mask shape ", conv_tgt_mask[self.kernel_size:, :].shape)
#         self.debugger.print("post_kernel_tgt_embed shape", post_kernel_tgt_embed.shape)

        # self-attention with convolution
        x = tgt_embed + self.conv_attention(
            query = tgt_embed,
            key=tgt_conv_embed,
            value=tgt_conv_embed,
            key_padding_mask=conv_tgt_padding_mask,
            attn_mask=conv_tgt_mask,
            need_weights = False
        )[0]


        x = self.norm1(x)
        self.debugger.print("norm1 x ", )
        self.debugger.print("x shape", x.shape)
        self.debugger.print("mem type", mem.dtype)
        # multi-headed attention with mem
        x = x + self.multi_head_attn(
            query = x,
            key = mem,
            value = mem,
            attn_mask = mem_mask,
            key_padding_mask = mem_padding_mask,
            need_weights = False
        )[0]
        x = self.norm2(x)
        self.debugger.print("norm2 x ", )

        # linear layer
        # x = self.linear2(self.activation(self.linear1(x)))
        # x = self.norm3(x)

        # transformer block ends

        self.debugger.print("x ", )
        return self.generator(x)


    def forward(self, batch):
        x, c, x_len, c_len = batch['x'], batch['c'], batch['x_len'], batch['c_len']
        B, L, _ = x.shape

        mem = self.encode(c)

        conv_tgt_len = self.l_in_to_l_out(L, self.kernel_size, self.stride, self.padding)

        if conv_tgt_len > 0: # x is not long enough sequentially for a kernel
            conv_tgt_mask = self.conv_attn_mask[:L, :conv_tgt_len]
            conv_tgt_padding_mask = self.convolution_padding_mask(conv_tgt_len, x_len).to(DEVICE)
            self.debugger.print("conv_tgt_mask shape", conv_tgt_mask.shape)
            self.debugger.print("conv_tgt_padding_mask shape", conv_tgt_padding_mask.shape)
            self.debugger.print("conv_tgt_padding_mask pad count", torch.sum(conv_tgt_padding_mask, dim=1))
        else:
            conv_tgt_mask = None
            conv_tgt_padding_mask = None

        mem_padding_mask = create_pad_mask(c, c_len).to(DEVICE)

        self.debugger.print("tgt shape", x.shape)
        self.debugger.print("mem_padding_mask shape", mem_padding_mask.shape)

        decoding = self.decode(x, mem, conv_tgt_mask, None, conv_tgt_padding_mask, mem_padding_mask)
        return decoding

    def loss_fn(self, decode_strokes, batch):
        tgt_strokes = batch['y']

        mask = create_pad_mask(decode_strokes, batch['x_len']).to(DEVICE)

        coord_loss = F.mse_loss(tgt_strokes[..., :2], decode_strokes[..., :2], reduction="none")
        self.debugger.print("coord_loss ", coord_loss)
        coord_mask = mask.unsqueeze(-1).expand(coord_loss.shape)
        coord_loss = coord_loss.masked_fill_(coord_mask, 0)
        self.debugger.print("coord_loss ", coord_loss)
        coord_loss = torch.sum(coord_loss) / torch.sum(coord_mask == False)

        self.debugger.print("coord_loss shape", coord_loss.shape)

        return coord_loss

    def convolution_attention_mask(self):
        mask = create_pad_mask_by_len(
            self.l_in_to_l_out(self.max_stroke_len, self.kernel_size, self.stride, self.padding),
            torch.tensor([self.l_in_to_l_out(i, self.kernel_size, self.stride, self.padding) for i in range(self.max_stroke_len)])
        )

#         mask[:, 0] = False # so that multiheaded attention don't have nan
        return mask

    def convolution_padding_mask(self, conv_tgt_len, tgt_len):
        self.debugger.print(
            "len",
            conv_tgt_len
        )

        return create_pad_mask_by_len(
            conv_tgt_len,
            torch.tensor([self.l_in_to_l_out_ceil(l, self.kernel_size, self.stride, self.padding) for l in tgt_len])
        )

    def l_in_to_l_out(self, l_in, kernel_size, stride, padding, dilation=1):
        return math.floor(
            (l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        )

    def l_in_to_l_out_ceil(self, l_in, kernel_size, stride, padding, dilation=1):
        return math.ceil(
            (l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        )

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)





class TFHW_GMX(nn.Module):
    def __init__(self) -> None:
        super(TFHW_GMX, self).__init__()

        self.src_embed = nn.Sequential(
            nn.Embedding(ALPHABET_SIZE, TF_D_MODEL),
            PositionalEncoding()
        )

        tf_encoder_layer = nn.TransformerEncoderLayer(
            d_model=TF_D_MODEL,
            nhead=TF_N_HEADS,
            dim_feedforward=TF_DIM_FEEDFORWARD,
            batch_first=True,
            dropout=TF_DROPOUT
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=tf_encoder_layer,
            num_layers=TF_ENC_LAYERS,
            norm=nn.LayerNorm(TF_D_MODEL)
        )

        self.tgt_embed = nn.Sequential(
            nn.Linear(D_STROKE, TF_D_MODEL),
            PositionalEncoding()
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=TF_D_MODEL,
            nhead=TF_N_HEADS,
            dim_feedforward=TF_DIM_FEEDFORWARD,
            dropout=TF_DROPOUT,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=tf_decoder_layer,
            num_layers=TF_DEC_LAYERS,
            norm=nn.LayerNorm(TF_D_MODEL)
        )

        self.z_e = nn.Linear(TF_D_MODEL, 1)
        self.z_pi = nn.Linear(TF_D_MODEL, MX_N_GAUSSIANS)
        self.z_mu1 = nn.Linear(TF_D_MODEL, MX_N_GAUSSIANS)
        self.z_mu2 = nn.Linear(TF_D_MODEL, MX_N_GAUSSIANS)
        self.z_sigma1 = nn.Linear(TF_D_MODEL, MX_N_GAUSSIANS)
        self.z_sigma2 = nn.Linear(TF_D_MODEL, MX_N_GAUSSIANS)
        self.z_rho = nn.Linear(TF_D_MODEL, MX_N_GAUSSIANS)

    def encode(self, batch_seq_char, batch_seq_char_len):
        encode_embed = self.src_embed(batch_seq_char)
        src_key_padding_mask = self.create_pad_mask(encode_embed, batch_seq_char_len)
        encoding = self.encoder(src=encode_embed, src_key_padding_mask=src_key_padding_mask)
        return encoding

    def decode(self, encoding, batch_seq_stroke, batch_seq_stroke_len):
        decode_embed = self.tgt_embed(batch_seq_stroke)
        tgt_key_padding_mask = self.create_pad_mask(decode_embed, batch_seq_stroke_len)
        tgt_mask = self.create_mask(decode_embed)

        out = self.decoder(
            tgt=decode_embed,
            tgt_mask=tgt_mask,
            memory=encoding,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # ===== Computing MDN =====
        es = self.z_e(out)
        # print("es shape ", es.shape) # -> torch.Size([sequence_length, batch, 1])
        es = 1 / (1 + torch.exp(es))
        # print("es shape", es.shape) # -> torch.Size([sequence_length, batch, 1])

        pis = self.z_pi(out)
        # print("pis shape ", pis.shape) # -> torch.Size([sequence_length, batch, n_gaussians])
        pis = torch.softmax(pis, 2)
        # print(pis.shape) # -> torch.Size([sequence_length, batch, n_gaussians])

        mu1s = self.z_mu1(out)
        mu2s = self.z_mu2(out)
        # print("mu shape :  ", mu1s.shape) # -> torch.Size([sequence_length, batch, n_gaussians])

        sigma1s = self.z_sigma1(out)
        sigma2s = self.z_sigma2(out)
        # print("sigmas shape ", sigma1s.shape) # -> torch.Size([sequence_length, batch, n_gaussians])
        sigma1s = torch.exp(sigma1s - self.bias)
        sigma2s = torch.exp(sigma2s - self.bias)
        # print(sigma1s.shape) # -> torch.Size([sequence_length, batch, n_gaussians])

        rhos = self.z_rho(out)
        rhos = torch.tanh(rhos)
        # print("rhos shape ", rhos.shape) # -> torch.Size([sequence_length, batch, n_gaussians])

        es = es.squeeze(2)
        # print("es shape ", es.shape) # -> torch.Size([sequence_length, batch])

        return es, pis, mu1s, mu2s, sigma1s, sigma2s, rhos

    def forward(self, batch):

        batch_seq_char = batch['c']
        batch_seq_char_len = batch['c_len']
        batch_seq_stroke = batch['x']
        batch_seq_stroke_len = batch['x_len']

        encoding = self.encode(batch_seq_char, batch_seq_char_len)
        return self.decode(encoding, batch_seq_stroke, batch_seq_stroke_len)


    def loss_fn(self, Pr, y, es):
        eps = float(np.finfo(np.float32).eps)
        loss1 = - torch.log(Pr + eps) # -> torch.Size([sequence_length, batch])
        bernouilli = torch.zeros_like(es) # -> torch.Size([sequence_length, batch])

        bernouilli = y[:, :, 2] * es + (1 - y[:, :, 2]) * (1 - es)

        loss2 = - torch.log(bernouilli + eps)
        loss = loss1 + loss2
        # print("loss shape", loss.shape) # -> torch.Size([sequence_length, batch])
        loss = torch.sum(loss, 0)
        # print("loss shape", loss.shape) # -> torch.Size([batch])

        return torch.mean(loss);

    def generate_sample(self, mu1, mu2, sigma1, sigma2, rho):
        mean = [mu1, mu2]
        cov = [[sigma1 ** 2, rho * sigma1 * sigma2], [rho * sigma1 * sigma2, sigma2 ** 2]]

        x = np.float32(np.random.multivariate_normal(mean, cov, 1))
        return torch.from_numpy(x)

    def generate_sequence(self, x0, words, bias):
        c0 = torch.tensor([alpha_to_num[c] for c in words]).unsqueeze(0).to(DEVICE)
        c0_len = torch.tensor([len(words)]).to(DEVICE)
        sequence = x0
        sample = x0
        sequence_length = c0.shape[1] * 25

        print("Generating sequence ...")
        self.bias = bias
        # f = FloatProgress(min=0, max=sequence_length)
        # display(f)

        encoding = self.encode(c0, c0_len)

        for i in range(sequence_length):
            es, pis, mu1s, mu2s, sigma1s, sigma2s, rhos = self.forward(sample, c0, True)
            # B x L x 3

            # Selecting a mixture
            pi_idx = np.random.choice(range(MX_N_GAUSSIANS), p=pis[-1, 0, :].detach().cpu().numpy())

            # taking last parameters from sequence corresponding to chosen gaussian
            mu1 = mu1s[-1, :, pi_idx].item()
            mu2 = mu2s[-1, :, pi_idx].item()
            sigma1 = sigma1s[-1, :, pi_idx].item()
            sigma2 = sigma2s[-1, :, pi_idx].item()
            rho = rhos[-1, :, pi_idx].item()

            prediction = self.generate_sample(mu1, mu2, sigma1, sigma2, rho)
            eos = torch.distributions.bernoulli.Bernoulli(torch.tensor([es[-1, :].item()])).sample()

            sample = torch.zeros_like(x0) # torch.Size([1, 1, 3])
            sample[0, 0, 0] = prediction[0, 0]
            sample[0, 0, 1] = prediction[0, 1]
            sample[0, 0, 2] = eos

            sequence = torch.cat((sequence, sample), 0) # torch.Size([sequence_length, 1, 3])

            # f.value += 1

        self.bias = 0

        return sequence.squeeze(1).detach().cpu().numpy()

    def gaussianMixture(self, y, pis, mu1s, mu2s, sigma1s, sigma2s, rhos):
        n_mixtures = pis.size(2)

        # Takes x1 and repeats it over the number of gaussian mixtures
        x1 = y[:,:, 0].repeat(n_mixtures, 1, 1).permute(1, 2, 0)
        # print("x1 shape ", x1.shape) # -> torch.Size([sequence_length, batch, n_gaussians])

        # first term of Z (eq 25)
        x1norm = ((x1 - mu1s) ** 2) / (sigma1s ** 2 )
        # print("x1norm shape ", x1.shape) # -> torch.Size([sequence_length, batch, n_gaussians])

        x2 = y[:,:, 1].repeat(n_mixtures, 1, 1).permute(1, 2, 0)
        # print("x2 shape ", x2.shape) # -> torch.Size([sequence_length, batch, n_gaussians])

        # second term of Z (eq 25)
        x2norm = ((x2 - mu2s) ** 2) / (sigma2s ** 2 )
        # print("x2norm shape ", x2.shape) # -> torch.Size([sequence_length, batch, n_gaussians])

        # third term of Z (eq 25)
        coxnorm = 2 * rhos * (x1 - mu1s) * (x2 - mu2s) / (sigma1s * sigma2s)

        # Computing Z (eq 25)
        Z = x1norm + x2norm - coxnorm

        # Gaussian bivariate (eq 24)
        N = torch.exp(-Z / (2 * (1 - rhos ** 2))) / (2 * np.pi * sigma1s * sigma2s * (1 - rhos ** 2) ** 0.5)
        # print("N shape ", N.shape) # -> torch.Size([sequence_length, batch, n_gaussians])

        # Pr is the result of eq 23 without the eos part
        Pr = pis * N
        # print("Pr shape ", Pr.shape) # -> torch.Size([sequence_length, batch, n_gaussians])
        Pr = torch.sum(Pr, dim=2)
        # print("Pr shape ", Pr.shape) # -> torch.Size([sequence_length, batch])

        Pr = Pr.to(DEVICE)

        return Pr



class TFHW_simple(nn.Module):

    def __init__(self) -> None:
        super(TFHW_simple, self).__init__()
        self.src_embed = nn.Sequential(
            nn.Embedding(ALPHABET_SIZE, TF_D_MODEL),
            PositionalEncoding()
        )

        self.tgt_embed = nn.Sequential(
            nn.Linear(D_STROKE, TF_D_MODEL),
            PositionalEncoding()
        )

        self.transformer = nn.Transformer(
            d_model=TF_D_MODEL,
            nhead=TF_N_HEADS,
            num_encoder_layers=TF_ENC_LAYERS,
            num_decoder_layers=TF_DEC_LAYERS,
            dim_feedforward=TF_DIM_FEEDFORWARD,
            dropout=TF_DROPOUT,
            batch_first=True
        )

        self.generator = nn.Linear(TF_D_MODEL, D_STROKE)

    def forward(self, batch):
        batch_seq_char = batch['c']
        batch_seq_char_len = batch['c_len']
        batch_seq_stroke = batch['x']
        batch_seq_stroke_len = batch['x_len']

        encode_embed = self.src_embed(batch_seq_char)
        src_key_padding_mask = create_pad_mask(encode_embed, batch_seq_char_len).to(DEVICE)

        decode_embed = self.tgt_embed(batch_seq_stroke)
        tgt_key_padding_mask = create_pad_mask(decode_embed, batch_seq_stroke_len).to(DEVICE)
        tgt_mask = self.transformer.generate_square_subsequent_mask(batch_seq_stroke.shape[1]).to(DEVICE)

        transformer_output = self.transformer(
            src=encode_embed,
            tgt=decode_embed,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask
        )

        if ADD_NOISE:
            transformer_output = transformer_output + self.noise.sample(transformer_output.size()).squeeze().to(DEVICE)

        decoding = self.generator(transformer_output)

        return decoding

    def loss_fn(self, decode_strokes, batch):
        tgt_strokes = batch['y']
        _, L, _ = tgt_strokes.shape

        mask = create_pad_mask(decode_strokes, batch['x_len']).to(DEVICE)

        coord_loss = F.mse_loss(decode_strokes[..., :2], tgt_strokes[..., :2], reduction="none")
        length_penalty_range = torch.arange(1, L + 1)
        length_penalty = torch.sqrt(length_penalty_range).unsqueeze(0).unsqueeze(-1).to(DEVICE)
        length_penalty = torch.min(length_penalty, torch.fill_(torch.zeros_like(length_penalty), 10))
        # so that the later strokes are penalized harder for not finding the style indicated by the previous strokes

        coord_mask = mask.unsqueeze(-1).expand(coord_loss.shape)
        coord_loss = coord_loss.masked_fill_(coord_mask, 0) * length_penalty
        coord_loss = torch.sum(coord_loss) / torch.sum(coord_mask == False)

        # reward for sparse points
        # x = batch['x']

        # reward for consistent distance between points



        return coord_loss


class TFHW(nn.Module):

    def __init__(self) -> None:
        super(TFHW, self).__init__()

        self.src_embed = nn.Sequential(
            nn.Embedding(ALPHABET_SIZE, TF_D_MODEL),
            PositionalEncoding()
        )

        tf_encoder_layer = nn.TransformerEncoderLayer(
            d_model=TF_D_MODEL,
            nhead=TF_N_HEADS,
            dim_feedforward=TF_DIM_FEEDFORWARD,
            batch_first=True,
            dropout=TF_DROPOUT
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=tf_encoder_layer,
            num_layers=TF_ENC_LAYERS,
            norm=nn.LayerNorm(TF_D_MODEL)
        )

        self.tgt_embed = nn.Sequential(
            nn.Linear(D_STROKE, TF_D_MODEL),
            PositionalEncoding()
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=TF_D_MODEL,
            nhead=TF_N_HEADS,
            dim_feedforward=TF_DIM_FEEDFORWARD,
            dropout=TF_DROPOUT,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=tf_decoder_layer,
            num_layers=TF_DEC_LAYERS,
            norm=nn.LayerNorm(TF_D_MODEL)
        )

        self.noise = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        self.generator = nn.Linear(TF_D_MODEL, D_STROKE)

    def forward(self, batch):

        batch_seq_char = batch['c']
        batch_seq_char_len = batch['c_len']
        batch_seq_stroke = batch['x']
        batch_seq_stroke_len = batch['x_len']

        encode_embed = self.src_embed(batch_seq_char)
        src_key_padding_mask = create_pad_mask(encode_embed, batch_seq_char_len).to(DEVICE)
        encoding = self.encoder(src=encode_embed, src_key_padding_mask=src_key_padding_mask)

        decode_embed = self.tgt_embed(batch_seq_stroke)
        tgt_key_padding_mask = create_pad_mask(decode_embed, batch_seq_stroke_len).to(DEVICE)
        tgt_mask = self.create_mask(decode_embed.shape[1])

        decoding = self.decoder(
            tgt=decode_embed,
            tgt_mask=tgt_mask,
            memory=encoding,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        if ADD_NOISE:
            decoding = decoding + self.noise.sample(decoding.size()).squeeze().to(DEVICE)

        decode_strokes = self.generator(decoding)
        decode_strokes[:, :, -1] = torch.sigmoid(decode_strokes[:, :, -1])

        return decode_strokes


    def loss_fn(self, decode_strokes, batch):
        tgt_strokes = batch['y']

        mask = create_pad_mask(decode_strokes, batch['x_len']).to(DEVICE)

        coord_loss = F.mse_loss(decode_strokes[..., :2], tgt_strokes[..., :2], reduction="none")
        coord_mask = mask.unsqueeze(-1).expand(coord_loss.shape)
        coord_loss = coord_loss.masked_fill_(coord_mask, 0)
        coord_loss = torch.sum(coord_loss)/ torch.sum(coord_loss == False)

        eos_loss = F.binary_cross_entropy(tgt_strokes[..., -1], decode_strokes[..., -1], reduction='none')
        eos_loss = eos_loss.masked_fill_(mask, 0)
        eos_loss = torch.sum(eos_loss) / torch.sum(mask == False)

        return coord_loss, eos_loss


    def create_mask(self, L) -> torch.tensor:
        return torch.triu(torch.full((L, L), float('-inf')), diagonal=1).to(DEVICE)
