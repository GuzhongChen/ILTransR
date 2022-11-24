import numpy as np
import pandas as pd
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split

def randomize_smile(sml):
    """Function that randomizes a SMILES sequnce. This was adapted from the
    implemetation of E. Bjerrum 2017, SMILES Enumeration as Data Augmentation
    for Neural Network Modeling of Molecules.
    Args:
        sml: SMILES sequnce to randomize.
    Return:
        randomized SMILES sequnce or
        nan if SMILES is not interpretable.
    """
    try:
        m = Chem.MolFromSmiles(sml)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        return Chem.MolToSmiles(nm, canonical=False, isomericSmiles=True)
    except:
        return float('nan')


def multi(sml,num):
    smiles_multi = pd.DataFrame(np.repeat(sml.values,num,axis=0))
    smiles_multi.columns = sml.columns
    return smiles_multi

def canonical_smile(sml):
    try:
        m = Chem.MolFromSmiles(sml)
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
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

def preprocess_list(smiles,mul=True):
    if mul:
        df = multi(smiles,10)
    else:
        df = pd.DataFrame(smiles)
    df["random_smiles"] = df["rdkit_canonical_smiles"].map(randomize_smile)   
    return df

if __name__ == "__main__":
    smiles = pd.read_csv("datasets/pubchem/pubchem_illike_100.csv")
    smiles["rdkit_canonical_smiles"] = smiles["SMILES"].map(canonical_smile)
    smiles_=smiles.dropna(axis=0,how='any')
    illike = preprocess_list(smiles_).dropna(axis=0,how='any')
    illike["source"] = illike["random_smiles"].map(no_split)
    illike["target"] = illike["rdkit_canonical_smiles"].map(no_split)

    X_train, X_test, y_train, y_test = train_test_split(illike['source'], illike['target'], test_size=100000, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=100000, random_state=1)
    pd.DataFrame(X_train).to_csv('datasets/pubchem/train.random_smiles',header=False,index=False)
    pd.DataFrame(y_train).to_csv('datasets/pubchem/train.rdkit_canonical_smiles',header=False,index=False)
    pd.DataFrame(X_val).to_csv('datasets/pubchem/val.random_smiles',header=False,index=False)
    pd.DataFrame(y_val).to_csv('datasets/pubchem/val.rdkit_canonical_smiles',header=False,index=False)
    pd.DataFrame(X_test).to_csv('datasets/pubchem/test.random_smiles',header=False,index=False)
    pd.DataFrame(y_test).to_csv('datasets/pubchem/test.rdkit_canonical_smiles',header=False,index=False)