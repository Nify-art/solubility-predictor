import streamlit as st
import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import shap

model = joblib.load("solubility_model.joblib")
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


def plain_language_summary(prediction, contributions):
    if prediction > -2:
        verdict = "fairly soluble in water"
    elif prediction > -4:
        verdict = "moderately soluble in water"
    else:
        verdict = "poorly soluble in water"

    sorted_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    top_feature, top_value = sorted_contributions[0]
    explanation = FEATURE_EXPLANATIONS[top_feature]
    direction = "high_means" if top_value < 0 else "low_means"
    reason = explanation[direction]

    return verdict, f"The biggest factor is its {explanation['name']}: {reason}."


st.title("Molecular Solubility Predictor")
st.write("Enter a molecule's SMILES string to predict its water solubility.")

smiles_input = st.text_input("SMILES string", value="CC(=O)Oc1ccccc1C(=O)O")

if st.button("Predict"):
    features = compute_features(smiles_input)

    if features is None:
        st.error("That doesn't look like a valid SMILES string. Try again.")
    else:
        prediction = model.predict(features)[0]
        shap_values = explainer(features)
        contributions = dict(zip(features.columns, shap_values.values[0]))

        verdict, reason = plain_language_summary(prediction, contributions)

        st.subheader(f"Predicted log solubility: {prediction:.3f}")
        st.write(f"This molecule is predicted to be **{verdict}**.")
        st.write(reason)

        st.subheader("Feature contributions")
        sorted_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, value in sorted_contributions:
            st.write(f"- **{feature}**: {value:+.3f}")