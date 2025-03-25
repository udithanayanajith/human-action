import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Load preprocessed data
data = np.load("preprocessed_medium_data.npz")
keypoints = data["keypoints"]
labels = data["labels"]

# Convert labels to integers
label_to_int = {label: i for i, label in enumerate(np.unique(labels))}
int_to_label = {v: k for k, v in label_to_int.items()}
labels = np.array([label_to_int[label] for label in labels])

# Print class distribution
unique, counts = np.unique(labels, return_counts=True)
print("Class distribution:", dict(zip([int_to_label[u] for u in unique], counts)))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(keypoints, labels, test_size=0.2, random_state=42, stratify=labels)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model (same as before)
class ActionRecognitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ActionRecognitionModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size // 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h_lstm1, _ = self.lstm1(x)
        h_lstm2, _ = self.lstm2(h_lstm1)
        h_lstm2 = h_lstm2[:, -1, :]
        out = self.fc(h_lstm2)
        return out

# Initialize model
input_size = X_train.shape[2]
hidden_size = 64
num_classes = len(label_to_int)
model = ActionRecognitionModel(input_size, hidden_size, num_classes)

# Compute class weights for imbalanced data
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float32)
print("Class weights:", class_weights)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

# Training loop
best_accuracy = 0
train_losses = []
val_accuracies = []

for epoch in range(100):
    model.train()
    epoch_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    train_losses.append(epoch_loss / len(train_loader))
    
    model.eval()
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    accuracy = 100 * correct / total
    val_accuracies.append(accuracy)
    scheduler.step(epoch_loss)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), "best_action_model.pth")
    
    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")
    print(classification_report(all_targets, all_predictions, target_names=label_to_int.keys(), zero_division=0))

# Plot training curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.savefig('training_curves.png')
plt.show()

print(f"Best accuracy: {best_accuracy:.2f}%")
print("Training completed. Best model saved as 'best_action_model.pth'")