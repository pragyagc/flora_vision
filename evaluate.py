import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CustomCNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#  CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_dir = "E:/FINAL_PROJECT/dataset_split_new/test" 
batch_size = 32
image_size = 128
checkpoint_path = "best_model.pth"

# TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

#  SAFE LOADER
def safe_loader(path):
    from PIL import Image
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except:
        print(f"Skipping corrupted image: {path}")
        return None

#  DATASET & DATALOADER
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform, loader=safe_loader)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
classes = test_dataset.classes
print("Classes:", classes)

#  LOAD MODEL 
model = CustomCNN(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# EVALUATION 
all_preds = []
all_labels = []
misclassified = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Record misclassified images
        for i in range(len(labels)):
            if predicted[i] != labels[i]:
                misclassified.append({
                    "image_path": test_dataset.samples[i][0],
                    "predicted": classes[predicted[i].item()],
                    "actual": classes[labels[i].item()]
                })

#  METRICS
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')

print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"Test Precision: {precision*100:.2f}%")
print(f"Test Recall: {recall*100:.2f}%")
print(f"Test F1 Score: {f1*100:.2f}%")
print(f"Total Misclassified Images: {len(misclassified)}")

#print first 10 misclassified images
for item in misclassified[:10]:
    print(f"Image: {item['image_path']}, Predicted: {item['predicted']}, Actual: {item['actual']}")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()    

