import torch
from model import CustomCNN
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from torchvision import transforms
import os

# ------------------ Settings ------------------
image_size = 128  # same as used in training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------ Classes ------------------
# Automatically get classes from folders if needed, or list manually
classes = [d for d in os.listdir("E:/8semproject/dataset_split/train") if os.path.isdir(os.path.join("E:/8semproject/dataset_split/train", d))]
classes.sort()  # ensure same order as training
print("Classes detected:", classes)

# ------------------ Load Model ------------------
num_classes = len(classes)
model = CustomCNN(num_classes=num_classes, input_size=(image_size, image_size)).to(device)

# Load model weights (state_dict)
checkpoint = torch.load("best_model.pth", map_location=device)
model = CustomCNN(num_classes=len(checkpoint["classes"])).to(device)
model.load_state_dict(checkpoint["model_state_dict"])


# ------------------ Prediction Function ------------------
def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0).to(device)  # add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return classes[predicted.item()]

# ------------------ Tkinter UI ------------------
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk
        
        pred = predict_image(file_path)
        result_label.config(text=f"Predicted Class: {pred}")

root = tk.Tk()
root.title("Plantopedia - Image Classifier")

# Upload button
btn = tk.Button(root, text="Upload Image", command=upload_image)
btn.pack(pady=10)

# Image panel
panel = tk.Label(root)
panel.pack(pady=10)

# Prediction label
result_label = tk.Label(root, text="Predicted Class: ", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
import torch
from model import CustomCNN
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from torchvision import transforms

# Settings
image_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load checkpoint
checkpoint = torch.load("best_model.pth", map_location=device)
classes = checkpoint["classes"]
print("Classes:", classes)

# Rebuild model
model = CustomCNN(num_classes=len(classes)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Prediction function
def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return classes[predicted.item()]

# Tkinter UI
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk

        pred = predict_image(file_path)
        result_label.config(text=f"Predicted Class: {pred}")

root = tk.Tk()
root.title("Plantopedia - Image Classifier")

btn = tk.Button(root, text="Upload Image", command=upload_image)
btn.pack(pady=10)

panel = tk.Label(root)
panel.pack(pady=10)

result_label = tk.Label(root, text="Predicted Class: ", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
