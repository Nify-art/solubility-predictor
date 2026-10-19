from rdkit import Chem
from rdkit.Chem import Descriptors

smiles = "Cc1occc1C(=O)Nc2ccccc2"  # Fenfuram, from your dataset
mol = Chem.MolFromSmiles(smiles)

molecular_weight = Descriptors.MolWt(mol)
num_h_donors = Descriptors.NumHDonors(mol)
num_h_acceptors = Descriptors.NumHAcceptors(mol)
logp = Descriptors.MolLogP(mol)

print(f"Molecular Weight: {molecular_weight}")
print(f"H-Bond Donors: {num_h_donors}")
print(f"H-Bond Acceptors: {num_h_acceptors}")
print(f"LogP: {logp}")