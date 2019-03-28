import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



class Discriminator(nn.Module):

    def __init__(self, embedding_dim, padding_size, f, conv_sizes, latent_size):
        """
            Input padding_size words, output shape (embedding_dim, padding_size).
            Conv2D kernel size (embedding_dim, h), output shape (1, padding_size-h+1).
            MaxPooling output shape (1, 1).
            f filters with m difference sizes, output shape fm.
            2 outputs: Linear output shape 1, Linear output shape latent_size.
        """
        super(Discriminator, self).__init__()
        
        self.conv = []
        self.maxpooling = []
        self.m = len(conv_sizes)
        for i in range(self.m):
            h = conv_sizes[i]
            self.conv.append(nn.Conv2d(in_channels=1, out_channels=f, kernel_size=(h, embedding_dim)))
            self.maxpooling.append(nn.MaxPool1d(kernel_size=(padding_size-h+1,)))
        self.linear = nn.Linear(in_features=f*self.m, out_features=1)
        self.recon = nn.Linear(in_features=f*self.m, out_features=latent_size)


    def forward(self, s, word_embeddings):
        # s is a sequence
        embed = word_embeddings(s)  # (batch_size, padding_size, embedding_dim)
        embed = torch.unsqueeze(input=embed, dim=1)  # (batch_size, 1, padding_size, embedding_dim)
        tmp_outputs = []
        for i in range(self.m):
            tmp = self.conv[i](embed)  # (batch_size, f, padding_size-h+1, 1)
            tmp = torch.squeeze(input=tmp, dim=3)  # (batch_size, f, padding_size-h+1)
            tmp_output = self.maxpooling[i](tmp)  # (batch_size, f, 1)
            tmp_output = torch.squeeze(input=tmp_output, dim=2)  # (batch_size, f)
            tmp_outputs.append(tmp_output)
        sent_vec = torch.cat(tensors=tmp_outputs, dim=1)  # (batch_size, fm)
        recon_z = self.recon(sent_vec)  # (batch_size, latent_size)
        pred = torch.sigmoid(self.linear(sent_vec))  # (batch_size, 1)
        return pred, recon_z, sent_vec




class Generator(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, latent_size, batch_size, padding_size):
        super(Generator, self).__init__()

        self.init = [nn.Linear(in_features=latent_size, out_features=hidden_size) for i in range(num_layers)]
        
        self.lstm = nn.LSTM(input_size=embedding_dim+latent_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.padding_size = padding_size


    def forward(self, x, z, h, c, word_embeddings):
        # x is a word because generator forward step by step
        if x is None:
            softmax_in = self.linear(h[-1])  # (batch_size, vocab_size)
            softmax_out = self.softmax(softmax_in)  # (batch_size, vocab_size)
            y = torch.argmax(input=softmax_out, dim=1, keepdim=True)  # (batch_size, 1)
            return y, h, c
        else:
            embed = word_embeddings(x)  # (batch_size, 1, embedding_dim)
            embed = torch.squeeze(embed, dim=1)  # (batch_size, embedding_dim)
            # z with (batch_size, latent_size)
            lstm_in = torch.cat(tensors=[embed, z], dim=1)  # (batch_size, embedding_dim+latent_size)
            # LSTM input need 3 dimensions
            lstm_in = torch.unsqueeze(lstm_in, dim=1)  # (batch_size, 1, embedding_dim+latent_size)
            lstm_out, (next_h, next_c) = self.lstm(lstm_in, (h, c))  # (batch_size, 1, hidden_size)
            lstm_out = torch.squeeze(lstm_out, dim=1)  # (batch_size, hidden_size)
            # if only one RNN step, next_h[-1] == lstm_out
            softmax_in = self.linear(lstm_out)  # (batch_size, vocab_size)
            softmax_out = self.softmax(softmax_in)  # (batch_size, vocab_size)
            y = torch.argmax(input=softmax_out, dim=1, keepdim=True)  # (batch_size, 1)
            return y, next_h, next_c


    def initHidden(self, z):
        # z with (batch_size, latent_size)
        h = [torch.unsqueeze(torch.tanh(self.init[i](z)), dim=0) for i in range(self.num_layers)]  # (num_layers, batch_size, hidden_size)
        h = torch.cat(h, dim=0)  # (num_layers, batch_size, hidden_size)
        c = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)  # (num_layers, batch_size, hidden_size)
        return h, c


    def generate(self, word_embeddings):
        # sample latent code, generate a fake sentence step by step
        batch_z = torch.FloatTensor(np.random.uniform(low=-0.01, high=0.01, size=(self.batch_size, self.latent_size)))
        g_input = None
        h, c = self.initHidden(batch_z)
        batch_fake_s = torch.zeros((self.batch_size, self.padding_size), dtype=torch.long)  # (batch_size, padding_size)
        for sent_step in range(self.padding_size):
            batch_y, h, c = self.forward(g_input, batch_z, h, c, word_embeddings)
            batch_fake_s[:, sent_step] = torch.squeeze(batch_y, dim=1)
            g_input = batch_y.detach()
        return batch_fake_s, batch_z



