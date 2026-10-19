import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

model = joblib.load("solubility_model.joblib")


def compute_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return pd.DataFrame([{
        "MolWt": Descriptors.MolWt(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
    }])


def predict_solubility(smiles):
    features = compute_features(smiles)
    if features is None:
        return None

    prediction = model.predict(features)[0]
    return prediction


if __name__ == "__main__":
    test_molecules = {
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    }

    for name, smiles in test_molecules.items():
        result = predict_solubility(smiles)
        print(f"{name}: predicted log solubility = {result:.3f}")