import gc
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm
from transformers import AdamW, AutoTokenizer, AutoModel, AutoConfig
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint
from sklearn.metrics import mean_squared_error, r2_score
from Task1 import MoLFormerWithRegressionHead, get_dataloaders

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
gc.collect()

# Define the full path to the model directory
model_path = "/home/neuronet_team146/Project_Files/scripts/mlm_finetuned_model"

# Load the fine-tuned model and tokenizer
mlm_model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
regression_model = MoLFormerWithRegressionHead(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("Model loaded successfully!")

# Tokenize and create DataLoader
def tokenize_data(smiles_list, tokenizer):
    encoded = tokenizer(smiles_list, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    return encoded["input_ids"], encoded["attention_mask"]

# Load the external dataset
external_data = pd.read_csv("/home/neuronet_team146/Project_Files/scripts/External_Dataset_for_Task2.csv")

# Tokenize SMILES strings
external_input_ids, external_attention_mask = tokenize_data(external_data["SMILES"].tolist(), tokenizer)

# Convert labels
external_labels = torch.tensor(external_data["Label"].values, dtype=torch.float).to(device)

# Create DataLoader for external dataset
external_dataset = TensorDataset(external_input_ids, external_attention_mask, external_labels)
# print('Length of external dataset: ',len(external_dataset))
external_dataloader = DataLoader(external_dataset, batch_size=1, shuffle=False)
# print('Length of external dataloader: ', len(external_dataloader))

# Compute gradients of loss w.r.t. model parameters
def compute_gradients(model, dataloader):
    print(f"Model is in training mode: {model.training}")
    model.train()
    grads = []
    criterion = nn.MSELoss()
    for batch in tqdm(dataloader, desc="Computing Gradients"):
        input_ids_batch, attention_mask_batch, labels_batch = batch
        input_ids = input_ids_batch.to(device)
        attention_mask = attention_mask_batch.to(device)
        labels = labels_batch.to(device).unsqueeze(0).requires_grad_(True)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = checkpoint(model, input_ids, attention_mask, use_reentrant=True)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)

        # DEBUGGING: Check if outputs require gradients
        # print(f"Outputs require grad: {outputs.requires_grad}")
        # print(f"Labels require grad: {labels.requires_grad}")

        loss_grads = grad(loss, model.parameters(), retain_graph=True, allow_unused=True) # added allow_unused=True
        grads.append(loss_grads)

        del input_ids, attention_mask, labels, outputs, loss, loss_grads
        torch.cuda.empty_cache()
        gc.collect()

    return grads

regression_model.train() 

# Compute gradients for the external dataset
external_grads = compute_gradients(regression_model, external_dataloader)
# print('Length of external grads: ', len(external_grads))

# Implementing LiSSA 
def lissa(grads, model, damping=0.01, scale=10, num_iter=200):
    device = next(model.parameters()).device
    h_estimate = [torch.zeros_like(p, device=device) for p in model.parameters()]
    filtered_grads = [g for g in grads[0] if g is not None]  

    if not filtered_grads:
        return [torch.zeros_like(p, device=device) for p in model.parameters()]

    for _ in range(num_iter):
        dot_products = []
        for g, h in zip(filtered_grads, h_estimate):
            dot_products.append(torch.sum(g.view(-1) * h.view(-1)))
        sum_dot_products = torch.stack(dot_products).sum()

        hvp = torch.autograd.grad(
            sum_dot_products,
            model.parameters(),
            retain_graph=True
        )
        with torch.no_grad():
            h_estimate = [
                h + (1 - damping) * he - s / scale
                for he, h, s in zip(h_estimate, hvp, filtered_grads)
            ]
    return h_estimate

def compute_influence_scores(external_grads, regression_model, device):
    influence_scores = []
    for ext_grad in tqdm(external_grads, desc="Computing Influence Scores"):
        filtered_grad = [g for g in ext_grad if g is not None]
        inverse_hvp = lissa([filtered_grad], regression_model)
        influence = sum((ihvp.cpu() * ext_g).sum().item() for ihvp, ext_g in zip(inverse_hvp, filtered_grad))
        influence_scores.append(influence)
    return influence_scores

# Get influence scores for external data points
influence_scores = compute_influence_scores(external_grads, regression_model, device)

# Add influence scores to the external dataset
external_data["Influence"] = influence_scores
# print('Length of influence : ', len(influence_scores))


# Save ranked dataset
external_data_sorted = external_data.sort_values(by="Influence", ascending=False)
external_data_sorted.to_csv("/home/neuronet_team146/Project_Files/scripts/External-Dataset_with_Influence.csv", index=False)

# print("Influence scores computed and saved!")


########################################### TASK 2b ##########################################################

k = 130 
print('k: ',k)
selected_samples = external_data_sorted.head(k)

selected_input_ids, selected_attention_mask = tokenize_data(selected_samples["SMILES"].tolist(), tokenizer)
selected_labels = torch.tensor(selected_samples["Label"].values, dtype=torch.float)

# Get the original training dataloader from Task1.py
train_dataloader, test_dataloader = get_dataloaders() 
train_dataset = train_dataloader.dataset

train_input_ids = train_dataset.tensors[0]
train_attention_mask = train_dataset.tensors[1]
train_labels = train_dataset.tensors[2]

# Combine datasets
combined_input_ids = torch.cat([train_input_ids, selected_input_ids], dim=0)
combined_attention_mask = torch.cat([train_attention_mask, selected_attention_mask], dim=0)
combined_labels = torch.cat([train_labels, selected_labels], dim=0)

combined_dataset = TensorDataset(combined_input_ids, combined_attention_mask, combined_labels)
combined_dataloader = DataLoader(combined_dataset, batch_size=16, shuffle=True)

# Fine-tuning
optimizer = AdamW(regression_model.parameters(), lr=5e-5, weight_decay=1e-2)
criterion = nn.MSELoss()
EPOCHS = 5

for epoch in range(EPOCHS):
    regression_model.train()
    running_loss = 0.0

    for batch in tqdm(combined_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        optimizer.zero_grad()
        outputs = regression_model(input_ids=input_ids, attention_mask=attention_mask).squeeze()
        loss = criterion(outputs, labels)


        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss / len(combined_dataloader):.4f}")

# Evaluation on test dataset.
regression_model.eval()
predictions = []
actual_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = regression_model(input_ids, attention_mask).squeeze()
        predictions.extend(outputs.cpu().numpy())
        actual_labels.extend(labels.cpu().numpy())

mse = mean_squared_error(actual_labels, predictions)
r2 = r2_score(actual_labels, predictions)

print(f'MSE: {mse}')
print(f'R2: {r2}')