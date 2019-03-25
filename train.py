import numpy as np
import pickle
from textGAN import *
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn


# ------------------------------- Load data & set parameters -----------------------------------

x = pickle.load(open('../data/data.p', 'rb'))    
train = x[0]
val = x[1]
# test = x[2]
wordtoix, ixtoword = x[9], x[10]
del x

vocab_size = len(wordtoix)
del wordtoix, ixtoword
padding_size = 0
for sent in train:
    if len(sent) > padding_size:
        padding_size = len(sent)
for sent in val:
    if len(sent) > padding_size:
        padding_size = len(sent)   
# for sent in test:
#     if len(sent) > padding_size:
#         padding_size = len(sent)

embedding_dim = 30
# f filters with m different sizes
f = 10
m = 5
conv_sizes = np.random.choice(np.arange(1, padding_size), size=5, replace=False)
input_size = embedding_dim + f * m
hidden_size = 20
num_layers = 1

batch_size = 32
lr_D = 0.01
lr_G = 0.01

# ---------------------------------- Dataset preparation --------------------------------------

# padding
train = [sent+[0]*(padding_size-len(sent)) for sent in train]
val = [sent+[0]*(padding_size-len(sent)) for sent in val]
# test = [sent+[0]*(padding_size-len(sent)) for sent in test]

# batch preparation
train_dataset = TensorDataset(torch.tensor(train))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(torch.tensor(val))
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
# test_dataset = TensorDataset(torch.tensor(test))
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# ----------------------------- Initialize GAN & optimizer--------------------------------------

word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
discriminator = Discriminator(vocab_size=vocab_size, embedding_dim=embedding_dim, padding_size=padding_size, f=f, conv_sizes=conv_sizes)
generator = Generator(vocab_size=vocab_size, embedding_dim=embedding_dim, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, fm=f*m, batch_size=batch_size)

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_G)


# -------------------------------------- Training ----------------------------------------------
for epoch in range(2):
    
    for d_step, batch in enumerate(train_loader):
        print ('\n d-step', d_step)
        # given input sentence, discriminator output prediction and sentence vector
        batch_x = batch[0]  # (batch_size, padding_size)
        batch_pred_x, batch_z = discriminator(batch_x, word_embeddings)

        for g_step in range(5):
            # given sentence vector, generator generate a fake sentence
            g_input = None
            h, c = generator.initHidden(batch_z)
            batch_fake_x = []
            for sent_step in range(padding_size):
                batch_y, h, c = generator(g_input, batch_z, h, c, word_embeddings)
                batch_fake_x.append(batch_y)
                g_input = batch_y.detach()
            batch_fake_x = torch.cat(batch_fake_x, dim=1)  # (batch_size, padding_size)
            # discriminator predict for the fake sentence
            batch_pred_fake_x, batch_fake_z = discriminator(batch_fake_x, word_embeddings)
            # compute generator loss and update
            z_mean = torch.mean(batch_z, dim=0)  # (batch_size, fm) => (fm)
            fake_z_mean = torch.mean(batch_fake_z, dim=0)  # (batch_size, fm) => (fm)
            G_loss = torch.sqrt(torch.mean((z_mean - fake_z_mean)**2))  # scalar
            optimizer_G.zero_grad()
            G_loss.backward(retain_graph=True)
            optimizer_G.step()
            print ('generator loss at step %d %.4f' % (g_step, G_loss))
        
        
        # compute discriminator loss and update
        D_loss = - torch.mean(torch.log(batch_pred_x) + torch.log(1. - batch_pred_fake_x))
        optimizer_D.zero_grad()
        D_loss.backward(retain_graph=True)
        optimizer_D.step()
        print ('discriminator loss at step %d %.4f' % (d_step, D_loss))

        # evaluate()

        
