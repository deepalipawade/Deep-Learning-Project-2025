# filter_and_save_external_data.py

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics.pairwise import cosine_similarity
import datasets

# Function to convert SMILES to Morgan fingerprint
def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    return None

# Function to compute similarity matrix and select relevant data
def filter_external_data(target_smiles, external_data, similarity_threshold=0.9):
    # Convert target dataset SMILES to fingerprints
    target_fps = [smiles_to_fingerprint(smiles) for smiles in target_smiles]
    
    # Convert external dataset SMILES to fingerprints
    external_fps = [smiles_to_fingerprint(smiles) for smiles in external_data['SMILES']]

    # Filter out None fingerprints
    valid_target_fps = [fp for fp in target_fps if fp is not None]
    valid_external_fps = [fp for fp in external_fps if fp is not None]

    if not valid_target_fps or not valid_external_fps:
        raise ValueError("No valid fingerprints found. Check SMILES conversion.")

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(valid_target_fps, valid_external_fps)

    # Select external data points that have high similarity to the target dataset
    selected_indices = np.where(similarity_matrix.max(axis=0) >= similarity_threshold)[0]
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
filtered_external_data = filter_external_data(target_smiles, external_data)
print(f"====================== Cosine filtered external data length : - ", len(filtered_external_data))

# Save filtered external data to a new CSV file
filtered_external_data.to_csv('/home/neuronet_team146/Project_Files/scripts/task3_data_selection/Cosine_Similarity_Filtered_Dataset.csv', index=False)

print(f"Filtered external data saved to '/home/neuronet_team146/Project_Files/scripts/task3_data_selection/Cosine_Similarity_Filtered_Dataset.csv'")
