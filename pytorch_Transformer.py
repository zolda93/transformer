import math
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Transformer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k,Multi30k
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from typing import List,Tuple,List,Iterable

device = 'cuda' if torch.cuda.is_available() else 'cpu'


multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"


#define special tokens and indices
UNK_IDX = 0
PAD_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

SPECIAL_TOKENS = ['<unk>', '<pad>', '<bos>', '<eos>']


source_language = 'de'
target_language = 'en'

# Place-holders

token_transform = {}
vocab_transform = {}


token_transform[source_language] = get_tokenizer('spacy',language='de_core_news_sm')
token_transform[target_language] = get_tokenizer('spacy', language='en_core_web_sm')


#helper function to yield list of tokens
def yield_tokens(data_iter:Iterable,language:str)->List[int]:
    language_index = {source_language:0,target_language:1}
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])


for ln in [source_language,target_language]:
    train_iter = Multi30k(split='train', language_pair=(source_language,target_language))
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter,ln),
                                                    min_freq=1,
                                                    specials=SPECIAL_TOKENS,
                                                    special_first=True)


for ln in [source_language,target_language]:
    vocab_transform[ln].set_default_index(UNK_IDX)

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]), 
                      torch.tensor(token_ids), 
                      torch.tensor([EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [source_language, target_language]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[source_language](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[target_language](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch



class PositionalEncoding(nn.Module):
    def __init__(self,
                embed_size:int,
                dropout:float,
                max_pos:int=5000)->None:

        super().__init__()

        den = torch.exp(-torch.arange(0,embed_size,2)*math.log(10000)/embed_size)
        pos = torch.arange(0,max_pos).reshape(max_pos,1)
        pos_embedding = torch.zeros((max_pos, embed_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout_layer = nn.Dropout(dropout)
        self.register_buffer('pos_embedding',pos_embedding)

    def forward(self,x:Tensor)->Tensor:
        return self.dropout_layer(x + self.pos_embedding[:x.size(0),:])


class Embedding(nn.Module):
    def __init__(self,vocab_size,embed_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.embed_size = embed_size

    def forward(self,tokens:Tensor)->Tensor:
        return self.embedding(tokens.long())*math.sqrt(self.embed_size)


class Translator(nn.Module):
    def __init__(self,
                num_encoder_layers:int,
                num_decoder_layers:int,
                embed_size:int,
                num_heads:int,
                source_vocab_size:int,
                target_vocab_size:int,
                dim_feedforward:int,
                dropout:float=0.1)->None:

        super().__init__()

        self.transformer = Transformer(d_model=embed_size,
                                        nhead=num_heads,
                                        num_encoder_layers=num_encoder_layers,
                                        num_decoder_layers=num_decoder_layers,
                                        dim_feedforward=dim_feedforward,
                                        dropout=dropout)

        self.generator = nn.Linear(embed_size,target_vocab_size)
        self.source_token_embed = Embedding(source_vocab_size,embed_size)
        self.target_token_embed = Embedding(target_vocab_size,embed_size)
        self.positional_encoding = PositionalEncoding(embed_size,dropout)

    def forward(self,
                source:Tensor,
                target:Tensor,
                source_mask:Tensor,
                target_mask:Tensor,
                source_padding_mask:Tensor,
                target_padding_mask:Tensor,
                memory_key_padding_mask:Tensor)->Tensor:

        source_embedding  = self.positional_encoding(self.source_token_embed(source))
        target_embedding  = self.positional_encoding(self.target_token_embed(target))
        outs = self.transformer(source_embedding,
                                target_embedding,
                                source_mask,
                                target_mask,
                                None,
                                source_padding_mask,
                                target_padding_mask,
                                memory_key_padding_mask)
        return self.generator(outs)


    def encode(self,source:Tensor,source_mask:Tensor)->Tensor:
        return self.transformer.encoder(self.positional_encoding(self.source_token_embed(source)),
                                                                source_mask)

    
    def decode(self,target:Tensor,memory:Tensor,target_mask:Tensor)->Tensor:
        return self.transformer.decoder(self.positional_encoding(self.target_token_embed(target)),memory,target_mask)



# During training, we need a subsequent word mask that will prevent model to look into the future words when making predictions. 
# We will also need masks to hide source and target padding tokens. Below, let's define a function that will take care of both.

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
                                                                


torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[source_language])
TGT_VOCAB_SIZE = len(vocab_transform[target_language])
EMBED_SIZE = 512
NUM_HEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 32
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

translator = Translator(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMBED_SIZE,
                                 NUM_HEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in translator.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

translator = translator.to(device)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(translator.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)



def train_epoch(model,optimizer)->float:
    model.train()
    losses = 0
    train_iter = Multi30k(split='train', language_pair=(source_language,target_language))
    train_dataloader = DataLoader(train_iter,batch_size=BATCH_SIZE,collate_fn=collate_fn)

    for source,target in train_dataloader:
        source = source.to(device)
        target = target.to(device)

        target_input = target[:-1,:]
        source_mask,target_mask,source_padding_mask,target_padding_mask = create_mask(source,target_input)
        logits = translator(source,target_input,source_mask,target_mask,source_padding_mask,target_padding_mask,source_padding_mask)
        optimizer.zero_grad()
        target_out = target[1:,:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), target_out.reshape(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()
    return losses



def evaluate(model):
    model.eval()
    losses = 0

    val_iter = Multi30k(split='valid', language_pair=(source_language,target_language))
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for source, target in val_dataloader:
        source = source.to(device)
        target = target.to(device)

        target_input = target[:-1, :]

        source_mask, target_mask, source_padding_mask, target_padding_mask = create_mask(source, target_input)

        logits = model(source, target_input, source_mask, target_mask,source_padding_mask, target_padding_mask, source_padding_mask)
        
        target_out = target[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), target_out.reshape(-1))
        losses += loss.item()

    return losses





def greedy_decode(model, source, source_mask, max_len, start_symbol):
    source = source.to(device)
    source_mask = source_mask.to(device)

    memory = model.encode(source, source_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        target_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, target_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(source.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys



# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, source_sentence: str):
    model.eval()
    source = text_transform[source_language](source_sentence).view(-1, 1)
    num_tokens = source.shape[0]
    source_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    target_tokens = greedy_decode(
        model,  source, source_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[target_language].lookup_tokens(list(target_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


