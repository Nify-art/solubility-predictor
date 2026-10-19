import pubchempy as pcp

name = "aspirin"
compounds = pcp.get_compounds(name, "name")

if compounds:
    smiles = compounds[0].connectivity_smiles
    print(f"{name} -> {smiles}")
else:
    print("Not found")