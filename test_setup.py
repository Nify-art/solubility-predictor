from rdkit import Chem

mol = Chem.MolFromSmiles("CCO")
print(mol)
print("RDKit is working!")
