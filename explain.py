import joblib
import pandas as pd
import shap
from rdkit import Chem
from rdkit.Chem import Descriptors

model = joblib.load("solubility_model.joblib")
data = pd.read_csv("processed_data.csv")
X = data.drop(columns=["measured log solubility in mols per litre"])


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


explainer = shap.TreeExplainer(model)

def explain_prediction(smiles):
    features = compute_features(smiles)
    if features is None:
        return None

    shap_values = explainer(features)
    prediction = model.predict(features)[0]

    contributions = dict(zip(features.columns, shap_values.values[0]))

    return {
        "prediction": prediction,
        "base_value": shap_values.base_values[0],
        "contributions": contributions,
    }


if __name__ == "__main__":
    test_molecules = {
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    }

    for name, smiles in test_molecules.items():
        result = explain_prediction(smiles)
        print(f"\n{name}: predicted = {result['prediction']:.3f}")
        sorted_contributions = sorted(
            result["contributions"].items(), key=lambda x: abs(x[1]), reverse=True
        )
        for feature, value in sorted_contributions:
            print(f"  {feature}: {value:+.3f}")