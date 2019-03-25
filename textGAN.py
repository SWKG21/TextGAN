import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



class Discriminator(nn.Module):

    def __init__(self, vocab_size, embedding_dim, padding_size, f, conv_sizes):
        """
            Input padding_size words, output shape (embedding_dim, padding_size).
            Conv2D kernel size (embedding_dim, h), output shape (1, padding_size-h+1).
            MaxPooling output shape (1, 1).
            f filters with m difference sizes, output shape fm.
            Linear output shape 1.
        """
        super(Discriminator, self).__init__()
        
        # self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.conv = []
        self.maxpooling = []
        self.m = len(conv_sizes)
        for i in range(self.m):
            h = conv_sizes[i]
            self.conv.append(nn.Conv2d(in_channels=1, out_channels=f, kernel_size=(h, embedding_dim)))
            self.maxpooling.append(nn.MaxPool1d(kernel_size=(padding_size-h+1,)))
        self.linear = nn.Linear(in_features=f*self.m, out_features=1)


    def forward(self, x, word_embeddings):
        embed = word_embeddings(x)  # (batch_size, padding_size, embedding_dim)
        embed = torch.unsqueeze(input=embed, dim=1)  # (batch_size, 1, padding_size, embedding_dim)
        tmp_outputs = []
        for i in range(self.m):
            tmp = self.conv[i](embed)  # (batch_size, f, padding_size-h+1, 1)
            tmp = torch.squeeze(input=tmp, dim=3)  # (batch_size, f, padding_size-h+1)
            tmp_output = self.maxpooling[i](tmp)  # (batch_size, f, 1)
            tmp_output = torch.squeeze(input=tmp_output, dim=2)  # (batch_size, f)
            tmp_outputs.append(tmp_output)
        sent_vec = torch.cat(tensors=tmp_outputs, dim=1)  # (batch_size, fm)
        output = torch.sigmoid(self.linear(sent_vec))  # (batch_size, 1)
        return output, sent_vec




class Generator(nn.Module):

    def __init__(self, vocab_size, embedding_dim, input_size, hidden_size, num_layers, fm, batch_size):
        super(Generator, self).__init__()

        # self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
        # self.softmax = nn.LogSoftmax(dim=2)

        self.init = [nn.Linear(in_features=fm, out_features=hidden_size) for i in range(num_layers)]
        # self.init = nn.Linear(in_features=fm, out_features=hidden_size)
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers


    def forward(self, x, z, h, c, word_embeddings):
        if x is None:
            softmax_in = self.linear(h[-1])  # (batch_size, vocab_size)
            # softmax_in = self.linear(torch.squeeze(h, dim=0))  # (batch_size, vocab_size)
            softmax_out = self.softmax(softmax_in)  # (batch_size, vocab_size)
            y = torch.argmax(input=softmax_out, dim=1, keepdim=True)  # (batch_size, 1)
            return y, h, c
        else:
            embed = word_embeddings(x)  # (batch_size, 1, embedding_dim)
            embed = torch.squeeze(embed, dim=1)  # (batch_size, embedding_dim)
            # z with (batch_size, fm)
            lstm_in = torch.cat(tensors=[embed, z], dim=1)  # (batch_size, embedding_dim+fm)
            # LSTM input need 3 dimensions
            lstm_in = torch.unsqueeze(lstm_in, dim=1)  # (batch_size, 1, embedding_dim+fm)
            lstm_out, (next_h, next_c) = self.lstm(lstm_in, (h, c))  # (batch_size, 1, hidden_size)
            lstm_out = torch.squeeze(lstm_out, dim=1)  # (batch_size, hidden_size)
            # if only one RNN step, next_h[-1] == lstm_out
            softmax_in = self.linear(lstm_out)  # (batch_size, vocab_size)
            softmax_out = self.softmax(softmax_in)  # (batch_size, vocab_size)
            y = torch.argmax(input=softmax_out, dim=1, keepdim=True)  # (batch_size, 1)
            return y, next_h, next_c


    # def forward(self, x, z, h, c, word_embeddings):
    #     embed = word_embeddings(x)  # (batch_size, padding_size, embedding_dim)
    #     # z with (batch_size, fm)
    #     z = torch.unsqueeze(z, dim=1) # (batch_size, 1, fm)
    #     z = torch.cat([z]*x.size()[1], dim=1)  # (batch_size, padding_size, fm)
    #     lstm_in = torch.cat([embed, z], dim=2)  # (batch_size, padding_size, embedding_dim+fm)
    #     lstm_out, (next_h, next_c) = self.lstm(lstm_in, (h, c))  # (batch_size, padding_size, hidden_size)
    #     softmax_in = self.linear(lstm_out)  # (batch_size, padding_size, vocab_size)
    #     softmax_out = self.softmax(softmax_in)  # (batch_size, padding_size, vocab_size)
    #     fake_x = torch.argmax(softmax_out, dim=2, keepdim=True)  # (batch_size, padding_size, 1)
    #     fake_x = torch.squeeze(fake_x, dim=2)  # (batch_size, padding_size)
    #     return fake_x, next_h, next_c



    def initHidden(self, z):
        h = [torch.unsqueeze(torch.tanh(self.init[i](z)), dim=0) for i in range(self.num_layers)]  # (num_layers, batch_size, hidden_size)
        h = torch.cat(h, dim=0)  # (num_layers, batch_size, hidden_size)
        # h = torch.unsqueeze(torch.tanh(self.init(z)), dim=0)  # (num_layers, batch_size, hidden_size)
        c = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)  # (num_layers, batch_size, hidden_size)
        return h, c







