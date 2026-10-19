import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
df = pd.read_csv(url)

def compute_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return pd.Series({
        "MolWt": Descriptors.MolWt(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
    })

features = df["smiles"].apply(compute_features)
data = pd.concat([features, df["measured log solubility in mols per litre"]], axis=1)
data = data.dropna()

print(data.shape)
print(data.head())

data.to_csv("processed_data.csv", index=False)
print("Saved to processed_data.csv")