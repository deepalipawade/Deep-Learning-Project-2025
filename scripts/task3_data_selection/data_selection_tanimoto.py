import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.metrics.pairwise import cosine_similarity
import datasets

# Function to convert SMILES to Morgan fingerprint
def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)  # Keep it as ExplicitBitVect
    return None


'''
In cheminformatics, the Tanimoto (or Jaccard) similarity is widely used for comparing binary molecular fingerprints. 
Compared to cosine similarity, Tanimoto similarity is especially suited for sparse, binary vectors (like Morgan fingerprints). 
It measures the ratio of the intersection to the union of bits set in the fingerprints, which often provides a more chemically meaningful similarity score.
'''
def filter_external_data_tanimoto(target_smiles, external_data, similarity_threshold=0.7):
    # Convert SMILES to fingerprints for target and external datasets
    target_fps = [smiles_to_fingerprint(smiles) for smiles in target_smiles]
    external_fps = [smiles_to_fingerprint(smiles) for smiles in external_data['SMILES']]

    # Filter out None values
    target_fps = [fp for fp in target_fps if fp is not None]
    external_fps = [fp for fp in external_fps if fp is not None]

    selected_indices = []
    for idx, ext_fp in enumerate(external_fps):
        # Compute the maximum Tanimoto similarity between an external fingerprint and all target fingerprints
        max_sim = max(DataStructs.TanimotoSimilarity(ext_fp, t_fp) for t_fp in target_fps)
        if max_sim >= similarity_threshold:
            selected_indices.append(idx)

    filtered_external_data = external_data.iloc[selected_indices]
    return filtered_external_data

# Load the dataset
DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
dataset = datasets.load_dataset(DATASET_PATH)
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)

# Load the external dataset
external_data = pd.read_csv('/home/neuronet_team146/Project_Files/scripts/External_Dataset_for_Task2.csv')

# Get the target SMILES from the training dataset
target_smiles = dataset['train']['SMILES']

# Filter external data based on similarity with target dataset
filtered_external_data = filter_external_data_tanimoto(target_smiles, external_data)
print(f"====================== Tanimoto filtered external data length : - ", len(filtered_external_data))

# Save filtered external data to a new CSV file
filtered_external_data.to_csv('/home/neuronet_team146/Project_Files/scripts/task3_data_selection/Tanimoto_Similarity_Filtered_Dataset.csv', index=False)

print(f"Filtered external data saved to '/home/neuronet_team146/Project_Files/scripts/task3_data_selection/Tanimoto_Similarity_Filtered_Dataset.csv'")
