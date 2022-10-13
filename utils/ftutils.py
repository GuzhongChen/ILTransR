import warnings
warnings.filterwarnings('ignore')
import os
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = cur_path[:cur_path.find('ILTransR')]+'ILTransR'
import random
import numpy as np
import mxnet as mx
import gluonnlp as nlp
from gluonnlp.model.transformer import get_transformer_encoder_decoder
nlp.utils.check_version('0.7.0')

np.random.seed(101)
random.seed(101)
mx.random.seed(10001)
ctx = mx.gpu(0)

# parameters for dataset
dataset = 'pubchem'
src_lang, tgt_lang = 'random_smiles', 'rdkit_canonical_smiles'
src_max_len, tgt_max_len = 100, 100

# parameters for model
num_units=128
hidden_size=1024
tf_dropout=0
epsilon=0.1
num_layers=3
num_heads=4
scaled=True
share_embed=True
embed_size=128
tie_weights=True
embed_initializer=None
magnitude = 3.0
lr_update_factor = 0.5
param_file = os.path.join(root_path,'pretraining/valid_best.params')

def _load_vocab(file_path, **kwargs):
    with open(file_path, 'r') as f:
        return nlp.Vocab.from_json(f.read())

src_vocab = _load_vocab(os.path.join(root_path,'datasets/pubchem/vocab.random_smiles.json'))
tgt_vocab = _load_vocab(os.path.join(root_path,'datasets/pubchem/vocab.rdkit_canonical_smiles.json'))

encoder, decoder, one_step_ahead_decoder = get_transformer_encoder_decoder(
    units=num_units,
    hidden_size=hidden_size,
    dropout=tf_dropout,
    num_layers=num_layers,
    num_heads=num_heads,
    max_src_length=src_max_len,
    max_tgt_length=tgt_max_len,
    scaled=scaled)
model = nlp.model.translation.NMTModel(src_vocab=src_vocab,
                 tgt_vocab=tgt_vocab,
                 encoder=encoder,
                 decoder=decoder,
                 one_step_ahead_decoder=one_step_ahead_decoder,
                 embed_size=num_units,
                 embed_initializer=None,
                 prefix='transformer_')

model.load_parameters(param_file,ctx=ctx)


from rdkit import Chem
def canonical_smile(sml):
    try:
        m = Chem.MolFromSmiles(sml)
        return Chem.MolToSmiles(m, canonical=True,isomericSmiles=False)
    except:
        return float('nan')

def no_split(sm):
    arr = []
    i = 0
    try:
        len(sm)
    except:
        print(sm)
    while i < len(sm)-1:
        arr.append(sm[i])
        i += 1
    if i == len(sm)-1:
        arr.append(sm[i])
    return ' '.join(arr)

length_clip = nlp.data.ClipSequence(100)
# Helper function to preprocess a single data point
def preprocess(data):
    # A token index or a list of token indices is
    # returned according to the vocabulary.
    src_sentence = src_vocab[data.split()]
    src_sentence.append(src_vocab[src_vocab.eos_token])
    src_npy = np.array(src_sentence, dtype=np.int32)
    src_nd = mx.nd.array(src_npy)
    return src_nd

# Helper function for getting the length
def get_length(x):
    return float(len(x.split(' ')))

