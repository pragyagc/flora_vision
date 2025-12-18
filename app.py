from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import os
from model import CustomCNN

app = Flask(__name__)

# ----------------- Model Setup -----------------
model = CustomCNN(num_classes=10)    # you have 10 classes
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device("cpu")))
model.eval()

# Class labels (same as training folder order)
classes = [
    'baganbilas', 'godawari', 'gulab', 'gulbahar', 'japakusum',
    'lalupatey', 'makhmali', 'sayapatri', 'sui ful', 'thulo_lwang'
]

# ----------------- PLANT DESCRIPTIONS -----------------
descriptions = {
    "baganbilas": "A colorful ornamental plant commonly found in Nepali gardens.",
    "godawari": "A  flowering plant widely used in rituals and decorations.",
    "gulab": "A fragrant rose species found in many colors, symbolizing love.",
    "gulbahar": "A daisy-like flower often used during Nepali festivals.",
    "japakusum": "Aflower used in worship and ayurvedic medicine.",
    "lalupatey": "A  plant commonly used as decoration in Nepali homes.",
    "makhmali": "A soft purple globe-shaped flower used for Tihar garlands.",
    "sayapatri": "The famous orange/yellow marigold used extensively in Tihar.",
    "sui ful": "A decorative flower with thin, needle-like petals.",
    "thulo_lwang": "A large aromatic flower used in rituals and cultural events."
}

# ----------------- Transform -----------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file selected")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No image selected")

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)

    prediction = classes[predicted.item()]
    description = descriptions[prediction]        # Get description
    confidence = confidence.item() * 100

    return render_template(
        'result.html',
        image_path=image_path,
        prediction=prediction,
        confidence=confidence,
        description=description
    )

if __name__ == '__main__':
    app.run(debug=True)
