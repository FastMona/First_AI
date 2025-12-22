# Digit detection program for MNIST model
# Loads trained model and predicts digits from image files

import torch
from PIL import Image
from torch import nn, load
from torchvision.transforms import ToTensor

# Define Image Classifier (must match training architecture)
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)        
        )
    
    def forward(self, x):
        return self.model(x)

def predict_image(image_path, model):
    """
    Predict the digit in an image file
    
    Args:
        image_path: Path to the image file (should be 28x28 grayscale)
        model: Trained ImageClassifier model
    
    Returns:
        prediction: Predicted digit (0-9)
        confidence: Confidence percentage
        all_scores: All class probabilities
    """
    # Load and convert image to tensor
    img = Image.open(image_path)
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output).item()
        
        # Calculate confidence scores
        probs = torch.softmax(output, dim=1)[0]
        confidence = probs[prediction].item()
    
    return prediction, confidence, probs

def main():
    print("="*60)
    print("MNIST Digit Detector")
    print("="*60)
    
    # Load the trained model
    print("\nLoading trained model from model_state.pth...")
    clf = ImageClassifier().to('cuda')
    
    try:
        with open('model_state.pth', 'rb') as f:
            clf.load_state_dict(load(f))
        clf.eval()
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: model_state.pth not found!")
        print("Please train the model first using torchnn.py")
        return
    
    # Get image filename from user
    print("\n" + "-"*60)
    image_path = input("Enter image filename (e.g., img_1.jpg): ")
    
    try:
        # Make prediction
        prediction, confidence, probs = predict_image(image_path, clf)
        
        # Display results
        print("\n" + "="*60)
        print(f"PREDICTION: {prediction}")
        print(f"CONFIDENCE: {confidence*100:.2f}%")
        print("="*60)
        
        print("\nAll class probabilities:")
        for digit in range(10):
            bar = "█" * int(probs[digit] * 50)
            print(f"  {digit}: {probs[digit]*100:5.2f}% {bar}")
        
    except FileNotFoundError:
        print(f"\nError: Image file '{image_path}' not found!")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
