import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import h5py
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import time

"""
This function converts a list of SMILES strings into a one-hot encoded format.
Each SMILES string is represented as a 3D numpy array where the dimensions 
correspond to the number of SMILES strings, the maximum length of the SMILES 
strings (120 in this case), and the size of the character set. The function 
initializes a zero array of the appropriate shape and then sets the corresponding 
indices to 1 based on the presence of characters in the SMILES strings.
"""
def one_hot_encode_smiles(smiles_list, charset):
    max_length = 120
    one_hot_encoded = np.zeros((len(smiles_list), max_length, len(charset)), dtype=np.int32)
    for i, smiles in enumerate(smiles_list):
        for j, char in enumerate(smiles):
            one_hot_encoded[i, j, charset.index(char)] = 1
    return one_hot_encoded.astype(np.float32)

"""
This utility function converts a list of character indices back into a SMILES 
string by mapping the indices to characters in the character set. This is useful 
for decoding the output of the model to a human-readable format.
"""
def decode_smiles_from_indexes(vec, charset):
    return "".join([charset[x] for x in vec]).strip()

def load_dataset(filename, split=True):
    h5f = h5py.File(filename, 'r')
    if split:
        data_train = h5f['data_train'][:]
    else:
        data_train = None
    data_test = h5f['data_test'][:]
    charset = [c.decode('utf-8') for c in h5f['charset'][:]]
    h5f.close()
    if split:
        return data_train, data_test, charset
    else:
        return data_test, charset

"""
This class defines the architecture of the Variational Autoencoder (VAE) model
. It consists of several convolutional layers for feature extraction, linear 
layers for dimensionality reduction and expansion, and a GRU layer for sequence 
modeling. The model is divided into three main parts: encoding, sampling, and decoding.
"""
class MolecularVAE(nn.Module):
    def __init__(self):
        super(MolecularVAE, self).__init__()
        self.conv_1 = nn.Conv1d(120, 9, kernel_size=9)
        self.conv_2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv_3 = nn.Conv1d(9, 10, kernel_size=11)
        self.linear_0 = nn.Linear(20, 435)
        self.linear_1 = nn.Linear(435, 292)
        self.linear_2 = nn.Linear(435, 292)
        self.linear_3 = nn.Linear(292, 292)
        self.gru = nn.GRU(292, 501, 3, batch_first=True)
        self.linear_4 = nn.Linear(501, 28)
        self.relu = nn.ReLU()

    """
    This method applies a series of convolutional and linear layers to the 
    input data to encode it into a latent representation. It outputs the mean 
    and log variance of the latent variables.
    """
    def encode(self, x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x = x.view(x.size(0), -1)
        x = F.selu(self.linear_0(x))
        return self.linear_1(x), self.linear_2(x)

    """
    This method implements the reparameterization to sample from the 
    latent space. It generates random noise and combines it with the mean 
    and log variance to create a sample from the latent space.
    """
    def sampling(self, z_mean, z_logvar):
        epsilon = torch.randn_like(z_logvar) * 0.01
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    """
    This method decodes the latent representation back into the original 
    input space. It applies a linear layer followed by a GRU layer and 
    finally a softmax activation to generate the output probabilities.
    """
    def decode(self, z):
        z = F.selu(self.linear_3(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 120, 1)
        output, hn = self.gru(z)
        out_reshape = output.contiguous().view(-1, output.size(-1))
        y0 = F.softmax(self.linear_4(out_reshape), dim=1)
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        return y

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        return self.decode(z), z_mean, z_logvar

"""
This function calculates the loss for the VAE. The loss consists of two parts:
the reconstruction loss, which measures how well the decoded output matches 
the original input, and the KL divergence, which regularizes the latent space 
to follow a standard normal distribution. The total loss is the sum of these 
two components.
"""
def vae_loss(x_decoded_mean, x, z_mean, z_logvar):
    xent_loss = F.binary_cross_entropy(x_decoded_mean, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return xent_loss + kl_loss

data_train, data_test, charset = load_dataset('processed.h5')
data_train = torch.utils.data.TensorDataset(torch.from_numpy(data_train))
train_loader = torch.utils.data.DataLoader(data_train, batch_size=250, shuffle=True)

def read_first_column_from_txt(file_path):
    first_column = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Ensure the line is not empty
                first_column.append(line.split()[0])
    return first_column


"""The training settings define the number of epochs, device (CPU or GPU),
model initialization, and optimizer. The train function runs the training 
loop for one epoch. It iterates over the batches of data, performs a 
forward pass, computes the loss, performs a backward pass, and updates 
the model parameters. The function also prints the training loss for each 
epoch to monitor the training progress."""

txt_file_path = 'small_mols.csv'  # Replace with your CSV file path
smiles_list = read_first_column_from_txt(txt_file_path)[0:100000]
charset = list(set(''.join(smiles_list)))
charset.sort()
data = one_hot_encode_smiles(smiles_list, charset)
num_samples = data.shape[0]
test_size = int(num_samples * 0.10)

# Randomly shuffle the indices
indices = np.arange(num_samples)
np.random.shuffle(indices)

# Split the indices for training and testing
test_indices = indices[:test_size]
train_indices = indices[test_size:]

# Split the data into training and testing sets
data_train = data[train_indices]
data_test = data[test_indices]

data_train = torch.utils.data.TensorDataset(torch.from_numpy(data_train))
train_loader = torch.utils.data.DataLoader(data_train, batch_size=500, shuffle=True)

data_test = torch.utils.data.TensorDataset(torch.from_numpy(data_test))
test_loader = torch.utils.data.DataLoader(data_test, batch_size=500, shuffle=True)

epochs = 2000
device = 'cuda'
model = MolecularVAE().to(device)
optimizer = optim.Adam(model.parameters())

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data,) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        output, mean, logvar = model(data)

        loss = vae_loss(output, data, mean, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f'Train Epoch: {epoch} \tLoss: {train_loss / len(train_loader.dataset):.6f}')
    return train_loss / len(train_loader.dataset)


for epoch in range(1, epochs + 1):
    train_loss = train(epoch)


torch.save(model, "autoencoder.model")