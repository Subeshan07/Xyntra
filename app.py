import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pennylane as qml
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import torchvision.models as models
import base64
from io import BytesIO

app = Flask(__name__)
app.secret_key = "quantum_lung_cancer_detection"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Quantum parameters (must match training params)
num_qubits = 6
num_layers = 6

# Quantum device configuration
dev = qml.device("default.qubit", wires=num_qubits)

# Quantum circuit definition (same as in training)
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(num_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

# QNode with parameter shapes
weight_shapes = {"weights": (num_layers, num_qubits, 3)}
qnode = qml.QNode(quantum_circuit, dev, interface="torch")

# Quantum Layer (same as in training)
class EnhancedQuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        scaled_input = x * self.scale
        return self.q_layer(scaled_input)

# Hybrid Model definition (matching the training model)
class QuantumHybridModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Feature extractor using ResNet50
        self.backbone = models.resnet50(pretrained=True)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Feature reduction for quantum processing
        self.dim_reduction = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_qubits)
        )
        
        # Quantum layer
        self.quantum_layer = EnhancedQuantumLayer()
        
        # Output layer
        self.classifier = nn.Sequential(
            nn.Linear(num_qubits, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        quantum_input = self.dim_reduction(features)
        quantum_output = self.quantum_layer(quantum_input)
        logits = self.classifier(quantum_output)
        return logits

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model
def load_model(model_path='best_quantum_lung_cancer_model.pth'):
    try:
        checkpoint = torch.load(model_path, map_location=device)
        class_names = checkpoint['class_names']
        num_classes = len(class_names)
        
        # Initialize the model with the correct number of classes
        model = QuantumHybridModel(num_classes).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Classes: {class_names}")
        
        return model, class_names
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# Load the model at startup
model, class_names = load_model()

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Make a prediction
def predict(image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        
        # Get probabilities and predictions
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class_idx = torch.argmax(probabilities).item()
        
        # Get class name and probability
        predicted_class = class_names[predicted_class_idx]
        confidence = probabilities[predicted_class_idx].item()
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probabilities, min(3, len(class_names)))
        top_predictions = [
            (class_names[idx], prob.item()) 
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        return predicted_class, confidence, top_predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        # Check if it's an image
        try:
            img = Image.open(file).convert('RGB')
            
            # Save the file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            img.save(filepath)
            
            # Preprocess and predict
            image_tensor = preprocess_image(img)
            predicted_class, confidence, top_predictions = predict(image_tensor)
            
            # Convert image to base64 for display
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Return results
            return render_template('results.html', 
                                  image_data=img_str,
                                  predicted_class=predicted_class, 
                                  confidence=confidence*100,
                                  top_predictions=top_predictions)
        
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(request.url)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        img = Image.open(file).convert('RGB')
        image_tensor = preprocess_image(img)
        predicted_class, confidence, top_predictions = predict(image_tensor)
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'top_predictions': [{'class': p[0], 'probability': p[1]} for p in top_predictions]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)