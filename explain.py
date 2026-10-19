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

FEATURE_EXPLANATIONS = {
    "LogP": {
        "name": "fat vs. water preference",
        "high_means": "the molecule prefers fat/oil over water, making it harder to dissolve",
        "low_means": "the molecule prefers water, making it easier to dissolve",
    },
    "TPSA": {
        "name": "surface polarity",
        "high_means": "the molecule's surface attracts water, helping it dissolve",
        "low_means": "the molecule's surface doesn't attract water much, making it harder to dissolve",
    },
    "MolWt": {
        "name": "molecular size",
        "high_means": "the molecule is large, which tends to make it harder to dissolve",
        "low_means": "the molecule is small, which tends to make it easier to dissolve",
    },
    "NumHDonors": {
        "name": "hydrogen bond donors",
        "high_means": "the molecule can form more bonds with water, helping it dissolve",
        "low_means": "the molecule forms fewer bonds with water, which can reduce how well it dissolves",
    },
    "NumHAcceptors": {
        "name": "hydrogen bond acceptors",
        "high_means": "the molecule can form more bonds with water, helping it dissolve",
        "low_means": "the molecule forms fewer bonds with water, which can reduce how well it dissolves",
    },
    "NumRotatableBonds": {
        "name": "molecular flexibility",
        "high_means": "the molecule is flexible, which can slightly affect how it dissolves",
        "low_means": "the molecule is rigid, which can slightly affect how it dissolves",
    },
}


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


def plain_language_summary(smiles, result):
    prediction = result["prediction"]
    contributions = result["contributions"]

    if prediction > -2:
        solubility_verdict = "fairly soluble in water"
    elif prediction > -4:
        solubility_verdict = "moderately soluble in water"
    else:
        solubility_verdict = "poorly soluble in water"

    sorted_contributions = sorted(
        contributions.items(), key=lambda x: abs(x[1]), reverse=True
    )
    top_feature, top_value = sorted_contributions[0]

    explanation = FEATURE_EXPLANATIONS[top_feature]
    direction = "high_means" if top_value < 0 else "low_means"
    reason = explanation[direction]

    summary = (
        f"This molecule is predicted to be {solubility_verdict}. "
        f"The biggest factor is its {explanation['name']}: {reason}."
    )

    return summary


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

        print(f"  Summary: {plain_language_summary(smiles, result)}")