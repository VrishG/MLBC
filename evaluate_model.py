import torch
from torch.utils.data import DataLoader
from data import YourDatasetClass  # Update this with the actual dataset class you're using
from model import AdvancedEnsembleModel  # Update this with the actual model class you're using
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Update these paths and variables as needed
model_checkpoint_path = 'saved\models\AdvancedEnsembleModel\0831_032038\checkpoint-epoch1.pth'
data_path = 'path_to_your_validation_data'  # Update this with the actual path
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load your dataset
valid_dataset = YourDatasetClass(data_path, train=False)  # Assuming train=False loads validation data
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Load your model
model = AdvancedEnsembleModel()
model.load_state_dict(torch.load(model_checkpoint_path))
model.to(device)
model.eval()

# Metrics initialization
all_preds = []
all_targets = []

with torch.no_grad():
    for data, target in valid_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        preds = torch.argmax(output, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_targets, all_preds)
precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

# Print the results
print(f'Validation Results - Epoch 1')
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1 * 100:.2f}%')
