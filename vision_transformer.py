# thanks to https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html

import numpy as np
from tqdm import tqdm ,trange

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor,Lambda
from torchvision.datasets import MNIST

np.random.seed(0)
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_patches(images:Tensor,n_patches:int)->Tensor:

    batch,channels,height,width = images.shape

    assert height== width,"need square images!!"

    patches = torch.zeros(batch,n_patches**2,height*width*channels//n_patches**2)
    patch_size = height//n_patches

    #generate patchs
    for idx,img in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patche=img[:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size]
                patches[idx,i*n_patches+j] = patche.flatten()
    return patches



class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,n_heads:int)->None:

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0,f"Can't divide dimension {d_model} into {n_heads} heads"

        self.d_head = self.d_model // self.n_heads
        self.Q = nn.ModuleList([nn.Linear(self.d_head,self.d_head) for _ in range(self.n_heads)])
        self.K = nn.ModuleList([nn.Linear(self.d_head,self.d_head) for _ in range(self.n_heads)])
        self.V = nn.ModuleList([nn.Linear(self.d_head,self.d_head) for _ in range(self.n_heads)])
        self.softmax = nn.Softmax(dim=-1)


    def forward(self,seqs:Tensor)->Tensor:
        
        result = []
        for seq in seqs:
            seq_result = []
            for head in range(self.n_heads):
                q,k,v = self.Q[head],self.K[head],self.V[head]
                sequence = seq[:,head*self.d_head:(head+1)*self.d_head]
                q,k,v = q(sequence),k(sequence),v(sequence)

                #attention
                attention = self.softmax(q@k.T/(self.d_head**0.5))
                seq_result.append(attention@v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r,dim=0) for r in result])



class ViTBlock(nn.Module):
    def __init__(self,embed_size:int,n_heads:int,ratio:int=4)->None:
        super().__init__()

        self.embed_size= embed_size
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(embed_size)
        self.attention = MultiHeadAttention(embed_size,n_heads)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feedforward = nn.Sequential(
                nn.Linear(embed_size,ratio*embed_size),
                nn.GELU(),
                nn.Linear(ratio*embed_size,embed_size))

    def forward(self,x:Tensor)->Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.feedforward(self.norm2(x))
        return x

from typing import Tuple

class ViT(nn.Module):
    def __init__(self,
                shape:Tuple[int],
                n_patches:int,
                n_blocks:int=2,
                embed_size:int=8,
                n_heads:int=2,
                out_dim:int=10,)->None:

        super().__init__()

        self.shape = shape
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.embed_size = embed_size

        # Input and patches sizes
        assert shape[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert shape[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (shape[1] / n_patches, shape[2] / n_patches)

        self.input_dim = int(shape[0]*self.patch_size[0]*self.patch_size[1])
        self.linear = nn.Linear(self.input_dim,self.embed_size)

        #learnable classification token
        self.class_token = nn.Parameter(torch.rand(1,self.embed_size))

        #positional encoding
        self.register_buffer('positional_encoding',get_positional_encoding(n_patches**2 + 1,embed_size),persistent=False)

        #Encoder/Decoder
        self.blocks = nn.ModuleList([ViTBlock(embed_size,n_heads) for _ in range(n_blocks)])

        #mlp
        self.mlp = nn.Sequential(
                nn.Linear(self.embed_size,out_dim),
                nn.Softmax(dim=-1)
                )


    def forward(self,images:Tensor)->Tensor:

        #generate patchs
        b,c,h,w = images.shape
        patches = generate_patches(images,self.n_patches).to(device)
        tokens = self.linear(patches)

        tokens = torch.cat((self.class_token.expand(b,1,-1),tokens),dim=1)

        output = tokens + self.positional_encoding.repeat(b,1,1)

        #transformer blocks
        for block in self.blocks:
            output = block(output)

        output = output[:,0]

        return self.mlp(output)


def get_positional_encoding(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result



def main()->None:
    

    training_data = MNIST(root='./mnist',train=True,download=True,transform=ToTensor())
    testing_data = MNIST(root='./mnist',train=False,download=False,transform=ToTensor())

    train_batch = DataLoader(training_data,batch_size=64,shuffle=True)
    test_batch = DataLoader(testing_data,batch_size=62,shuffle=False)

    model = ViT((1,28,28),n_patches=7,n_blocks=7,embed_size=8,n_heads=2,out_dim=10).to(device)

    epochs = 5
    lr = 0.005

    optimizer = Adam(model.parameters(),lr=lr)
    loss_fn = CrossEntropyLoss()

    for epoch in trange(epochs,desc='Training'):
        train_loss = 0.0
        for batch in tqdm(train_batch,desc=f"Epoch{epoch+1} in training",leave=False):
            x,y = batch
            x,y = x.to(device),y.to(device)
            y_hat = model(x)
            loss = loss_fn(y_hat,y)
            train_loss += loss.detach().cpu().item() / len(train_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs} loss: {train_loss:.2f}")

    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_batch, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_batch)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")














