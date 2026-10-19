# Molecular Solubility Predictor

A machine learning pipeline that predicts the water solubility of a molecule from its chemical structure (SMILES notation), using RDKit for feature extraction and a Random Forest model trained on the ESOL dataset. Includes an explainability layer so predictions aren't just a number, you can see exactly which molecular properties drove each one.

## What it does

Given a molecule's SMILES string, this project predicts its log solubility in water, a key early-stage filter in real drug discovery, by:

1. Parsing the molecule's structure with RDKit
2. Extracting 6 chemical descriptors (molecular weight, LogP, polar surface area, H-bond donors/acceptors, rotatable bonds)
3. Feeding those into a Random Forest model trained on ~1,100 real molecules with measured solubility values
4. Using SHAP to explain how each descriptor contributed to that specific prediction

## Setup

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

## Usage

Train the model (also saves it to disk):

python train_model.py

Predict solubility for new molecules:

python predict.py

Get a feature-by-feature explanation for a prediction:

python explain.py

## Results

- Linear Regression: R² = 0.742
- Random Forest: R² = 0.854
- Most influential feature overall: LogP (82% of model importance), consistent with known chemistry, LogP is a classic predictor of water solubility

## Example predictions

| Molecule | Predicted log solubility |
|----------|--------------------------|
| Caffeine | -1.075 |
| Aspirin | -1.887 |
| Ibuprofen | -3.218 |

Ordering matches known real-world solubility behavior of these drugs.

## Explainability

Beyond just predicting a number, this project uses SHAP (SHapley Additive exPlanations) to break down *why* a molecule got its predicted solubility, showing exactly how much each feature pushed the prediction up or down relative to the average.

For example:

- **Ibuprofen** (least soluble of the three): LogP contributes **-0.333**, the dominant factor pulling its solubility down. This lines up with real chemistry, ibuprofen has high fat-solubility, which is a well-known pharmaceutical formulation challenge.
- **Caffeine** (most soluble of the three): LogP contributes **+2.217**, the dominant factor pushing its solubility up, consistent with caffeine's known water-solubility.

This sign-flip in LogP's contribution across molecules shows the model isn't just repeating one fixed rule, it's responding differently depending on each molecule's actual chemistry, and in a way that checks out against known pharmaceutical knowledge.

Run:

python explain.py

## Dataset

[ESOL (Delaney) solubility dataset](https://pubs.acs.org/doi/10.1021/ci034243x), ~1,100 compounds with measured aqueous solubility.

## Project structure

solubility-predictor/
├── train_model.py        # Loads data, trains Linear Regression + Random Forest, saves the model
├── predict.py             # Loads the saved model, predicts solubility for new SMILES
├── explain.py              # Loads the saved model, explains predictions using SHAP
├── build_features.py    # Converts raw SMILES into molecular descriptors
├── get_data.py             # Fetches the ESOL dataset
├── processed_data.csv  # Processed dataset with computed features
├── solubility_model.joblib  # Trained model, saved to disk
├── requirements.txt
└── README.md