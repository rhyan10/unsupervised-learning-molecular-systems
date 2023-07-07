import pybel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import ase.io
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

class MolecularVAE(nn.Module):
    def __init__(self):
        super(MolecularVAE, self).__init__()

        self.conv_1 = nn.Conv1d(35, 20, kernel_size=5)
        self.conv_2 = nn.Conv1d(20, 15, kernel_size=5)
        self.conv_3 = nn.Conv1d(15, 9, kernel_size=5)
        self.linear_0 = nn.Linear(90, 200)
        self.linear_1 = nn.Linear(200, 200)
        self.linear_2 = nn.Linear(200, 200)

        self.linear_3 = nn.Linear(200, 200)
        self.gru = nn.GRU(200, 300, 3, batch_first=True)
        self.linear_4 = nn.Linear(300, 22)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def encode(self, x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x = x.view(x.size(0), -1)
        x = F.selu(self.linear_0(x))
        return self.linear_1(x), self.linear_2(x)

    def sampling(self, z_mean, z_logvar):
        epsilon = 1e-2 * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def decode(self, z):
        z = F.selu(self.linear_3(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 35, 1)
        output, hn = self.gru(z)
        out_reshape = output.contiguous().view(-1, output.size(-1))
        y0 = F.softmax(self.linear_4(out_reshape), dim=1)
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        return y

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        return self.decode(z), z_mean, z_logvar

def vae_loss(x_decoded_mean, x, z_mean, z_logvar):
    xent_loss = F.binary_cross_entropy(x_decoded_mean, x, size_average=False)
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return xent_loss + kl_loss

def smiles_to_hot(smiles, max_len, char_indices, nchars):
    X = np.zeros((len(smiles), max_len, nchars), dtype=np.float32)
    for i, smile in tqdm(enumerate(smiles)):
        for t, char in enumerate(smile):
                X[i, t, char_indices[char]] = 1
    return X

#Load in XYZ geomtries with XYZ2MOL module in RDKit into RDKit mol object

db = ase.io.read('qm7_scaled.xyz',':')
# for i, mol in enumerate(db):
#     ase.io.write('temp/mol'+str(i)+'.xyz', mol)

all_chars = []
max_len = 35

file_path = "smiles.txt"
all_smiles_strings = []
with open(file_path, "w") as file:
    for i in tqdm(range(len(db))):
        mol = next(pybel.readfile("xyz", 'temp/mol'+str(i)+'.xyz'))
        smi = mol.write(format="smi")
        smi = smi.split()[0].strip()
        file.write(smi)
        file.write("\n")
        characters = [char for char in smi]
        while len(characters) < max_len:
            characters.append(" ")
        all_chars.append(characters)
        all_smiles_strings.append(smi)

#unique_characters = set(all_chars)
char_indices = {'S':0, '2':1, 'O':2, '[':3, 'o':4, '\\':5, 'c':6, '3':7, 'C':8, ')':9, 's':10, 'N':11, '(':12, 'H':13, ']':14, '#':15, 'n':16, '1':17, '@':18, '/':19, '=':20, ' ':21}
one_hot_embeddings = smiles_to_hot(all_chars, max_len, char_indices, 22)
data_train = one_hot_embeddings[:6000]
data_test = one_hot_embeddings[6000:]
data_train = torch.utils.data.TensorDataset(torch.from_numpy(data_train))
data_test = torch.utils.data.TensorDataset(torch.from_numpy(data_test))

train_loader = torch.utils.data.DataLoader(data_train, batch_size=50, shuffle=True)

model = MolecularVAE()
optimizer = optim.Adam(model.parameters())

num_epoch = 300

for epoch in range(num_epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
            data = data[0]
            optimizer.zero_grad()
            output, mean, logvar = model(data)
            loss = vae_loss(output, data, mean, logvar)
            loss.backward()
            print("Loss: " +str(loss))
            train_loss += loss
            optimizer.step()
    #print('train', train_loss / len(train_loader.dataset))





