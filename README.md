# Molecular Solubility Predictor

A machine learning pipeline that predicts the water solubility of a molecule from its chemical structure (SMILES notation), using RDKit for feature extraction and a Random Forest model trained on the ESOL dataset.

## What it does

Given a molecule's SMILES string, this project predicts its log solubility in water, a key early-stage filter in real drug discovery, by:

1. Parsing the molecule's structure with RDKit
2. Extracting 6 chemical descriptors (molecular weight, LogP, polar surface area, H-bond donors/acceptors, rotatable bonds)
3. Feeding those into a Random Forest model trained on ~1,100 real molecules with measured solubility values

## Setup

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

## Usage

Train the model (also saves it to disk):

python train_model.py

Predict solubility for new molecules:

python predict.py

## Results

- Linear Regression: R² = 0.742
- Random Forest: R² = 0.854
- Most influential feature: LogP (82% of model importance), consistent with known chemistry, LogP is a classic predictor of water solubility

## Example predictions

| Molecule | Predicted log solubility |
|----------|--------------------------|
| Caffeine | -1.075 |
| Aspirin | -1.887 |
| Ibuprofen | -3.218 |

Ordering matches known real-world solubility behavior of these drugs.

## Dataset

[ESOL (Delaney) solubility dataset](https://pubs.acs.org/doi/10.1021/ci034243x), ~1,100 compounds with measured aqueous solubility.