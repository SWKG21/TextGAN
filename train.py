import numpy as np
import pickle
from textGAN import *
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from utils import MMD


# ------------------------------- Load data & set parameters -----------------------------------

data = pickle.load(open('../data/data.p', 'rb'))    
train = data[0]
val = data[1]
# test = data[2]
wordtoix, ixtoword = data[9], data[10]
del data

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

# generator parameters
latent_size = 10
hidden_size = 20
num_layers = 2

batch_size = 2
lr_D = 0.01
lr_G = 0.01

# ---------------------------------- Dataset preparation --------------------------------------

# padding
train = [sent+[0]*(padding_size-len(sent)) for sent in train]
val = [sent+[0]*(padding_size-len(sent)) for sent in val]
# test = [sent+[0]*(padding_size-len(sent)) for sent in test]

# batch preparation
train_dataset = TensorDataset(torch.tensor(train))
gen_train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
disc_train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
gen_train_loader_iter = iter(gen_train_loader)
disc_train_loader_iter = iter(disc_train_loader)

val_dataset = TensorDataset(torch.tensor(val))
gen_val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
disc_val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

# test_dataset = TensorDataset(torch.tensor(test))
# gen_test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
# disc_test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# ----------------------------- Initialize GAN & optimizer--------------------------------------

word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
discriminator = Discriminator(embedding_dim=embedding_dim, padding_size=padding_size, f=f, conv_sizes=conv_sizes, latent_size=latent_size)
generator = Generator(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, latent_size=latent_size, batch_size=batch_size, padding_size=padding_size)

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_G)
loss = nn.MSELoss()

# -------------------------------------- Training ----------------------------------------------
for epoch in range(10):
    print ('\nepoch', epoch)
    for g_step in range(5):
        # generator generate a batch of fake sentences
        batch_fake_s, _ = generator.generate(word_embeddings)
        # load s        
        batch_s = next(gen_train_loader_iter)[0]
        # discriminator predict for both real s and the fake sentence
        _, _, batch_fake_s_vec = discriminator(batch_fake_s, word_embeddings)
        _, _, batch_s_vec = discriminator(batch_s, word_embeddings)
        # compute generator loss and update
        G_loss = MMD(batch_fake_s_vec, batch_s_vec)  # minimize MMD (f and f_tilde to be similar)
        optimizer_G.zero_grad()
        G_loss.backward(retain_graph=True)
        optimizer_G.step()
        print ('generator loss at step %d %.4f' % (g_step, G_loss))
        
    for d_step in range(1):
        # generator generate a batch of fake sentences
        batch_fake_s, batch_z = generator.generate(word_embeddings)
        # load s        
        batch_s = next(disc_train_loader_iter)[0]
        # discriminator predict for both real s and the fake sentence
        batch_pred_fake_s, batch_recon_z, batch_fake_s_vec = discriminator(batch_fake_s, word_embeddings)
        batch_pred_s, _, batch_s_vec = discriminator(batch_s, word_embeddings)
        # compute discriminator loss and update
        gan_loss = - torch.mean(torch.log(batch_pred_s) - torch.log(1. - batch_pred_fake_s))  # minimize gan_loss (f and f_tilde to be discriminative)
        mmd_loss = MMD(batch_fake_s_vec, batch_s_vec)  # maximize MMD (f and f_tilde to be challenging)
        recon_loss = torch.mean(torch.norm(input=batch_recon_z-batch_z, p=2, dim=1))  # minimize recon_loss (f and f_tilde to be representative)
        D_loss = gan_loss - 0.1 * mmd_loss + 0.1 * recon_loss
        optimizer_D.zero_grad()
        D_loss.backward(retain_graph=True)
        optimizer_D.step()
        print ('discriminator loss at step %d %.4f' % (d_step, D_loss))


        # evaluate()

        
# generator loss does not change
# not following reasons: generate step by step; loss function; 