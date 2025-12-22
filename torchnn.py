# first attempt at building a convolutional neural network in PyTorch 
# this is as an image classifier for MNIST dataset

# Import torch to access functions like torch.argmax, torch.no_grad, torch.max
import torch
from PIL import Image
# Import specific components from torch to avoid typing torch.nn, torch.save, etc.
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Get data
# Download MNIST dataset (28x28 grayscale images of handwritten digits 0-9)
# ToTensor() converts PIL images to PyTorch tensors and normalizes pixel values to [0,1]
train = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
test = datasets.MNIST(root='data', train=False, download=True, transform=ToTensor())

# DataLoader wraps datasets to provide batching and shuffling
# batch_size=32 processes 32 images at once for efficient GPU computation
# shuffle=True randomizes training order to prevent learning patterns in data order
# shuffle=False for test to ensure consistent evaluation across runs
train_loader = DataLoader(train, batch_size=32, shuffle=True)
test_loader = DataLoader(test, batch_size=32, shuffle=False)

# Define Image Classifier and Neural Network
# Inheriting from nn.Module gives us PyTorch's neural network functionality
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Using Sequential to create a pipeline where output of one layer feeds into the next
        # This keeps the architecture clean for simple forward-flowing networks
        self.model = nn.Sequential(
            # First Conv2d: 1 input channel (grayscale), 32 output feature maps, 3x3 kernel
            # WHY: Start with 32 filters to detect basic patterns (edges, curves) in images
            nn.Conv2d(1, 32, (3,3)),
            # ReLU activation introduces non-linearity so network can learn complex patterns
            # WHY: Without activation functions, stacking layers would just be linear algebra
            nn.ReLU(),
            # Second Conv2d: 32 inputs from previous layer, 64 outputs
            # WHY: Double the filters to learn more complex combinations of simple patterns
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            # Third Conv2d: Keep 64 filters to learn even higher-level features
            # WHY: More layers = more abstraction (simple edges → digit parts → full digits)
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            # Flatten converts 2D feature maps into 1D vector for the final classifier
            # WHY: Dense layers expect 1D input, not 2D spatial feature maps
            nn.Flatten(),
            # Final Linear layer: (28-6)*(28-6) comes from 3 conv layers shrinking 28x28 by 2 pixels each
            # 10 outputs = one score for each digit (0-9)
            # WHY: Output layer size matches number of classes we're predicting
            nn.Linear(64*(28-6)*(28-6), 10)        
        )
    
    # forward defines how data flows through the network during prediction
    def forward(self, x):
        return self.model(x)

# Instantiate model, define loss function and optimizer
# Move model to GPU ('cuda') for faster training - GPUs excel at parallel matrix operations
# WHY: Training on GPU can be 10-100x faster than CPU for neural networks
clf = ImageClassifier().to('cuda')

# Adam optimizer adjusts model weights based on gradients during training
# lr=1e-3 (0.001) is the learning rate - how big of steps we take when updating weights
# WHY: Adam adapts learning rates per parameter, works well for most problems without tuning
opt = Adam(clf.parameters(), lr=1e-3)

# CrossEntropyLoss combines softmax and negative log likelihood
# WHY: Standard loss for multi-class classification - penalizes wrong predictions more heavily
loss_fn = nn.CrossEntropyLoss()

# Training loop
if __name__ == "__main__":
    # Load previously trained weights to test the model without retraining
    # WHY: Saves time - can use an already trained model for predictions
    with open('model_state.pth', 'rb') as f:
        clf.load_state_dict(load(f))

    # Test the model on a custom image
    img = Image.open('img_1.jpg')
    # ToTensor() converts to tensor, unsqueeze(0) adds batch dimension (model expects batches)
    # .to('cuda') moves tensor to GPU to match model's location
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')
    # argmax finds which digit (0-9) has the highest prediction score
    print(torch.argmax(clf(img_tensor)))


    # TRAINING CODE (currently commented out)
    # Uncomment to train the model from scratch for 10 epochs
    # for epoch in range(10):
    #     # Training phase
    #     # Set model to training mode - enables dropout, batch norm updates, etc.
    #     # WHY: Different behavior needed during training vs testing
    #     clf.train()
    #     train_loss = 0.0
    #     
    #     # Loop through all training batches
    #     for batch in train_loader:
    #         X, y = batch  # X = images, y = labels (0-9)
    #         # Move batch to GPU to match model
    #         X, y = X.to('cuda'), y.to('cuda')
    #         
    #         # Forward pass: get model predictions
    #         yhat = clf(X)
    #         # Calculate how wrong predictions are
    #         loss = loss_fn(yhat, y)
    #         
    #         # Backpropagation: calculate gradients and update weights
    #         # WHY: zero_grad() clears old gradients (PyTorch accumulates by default)
    #         opt.zero_grad()
    #         # Compute gradients of loss with respect to all parameters
    #         loss.backward()
    #         # Update weights using computed gradients
    #         opt.step()
    #         
    #         train_loss += loss.item()
    #     
    #     # Average loss across all batches for this epoch
    #     train_loss /= len(train_loader)
    #     
    #     # Validation/Test phase
    #     # Set model to evaluation mode - disables dropout, uses running stats for batch norm
    #     # WHY: We want consistent, deterministic predictions during testing
    #     clf.eval()
    #     test_loss = 0.0
    #     correct = 0
    #     total = 0
    #     
    #     # no_grad() disables gradient computation - saves memory and speeds up testing
    #     # WHY: We don't need gradients during evaluation, only during training
    #     with torch.no_grad():
    #         for batch in test_loader:
    #             X, y = batch
    #             X, y = X.to('cuda'), y.to('cuda')
    #             yhat = clf(X)
    #             loss = loss_fn(yhat, y)
    #             test_loss += loss.item()
    #             
    #             # Calculate accuracy
    #             # torch.max returns (values, indices) - we only need indices (predictions)
    #             _, predicted = torch.max(yhat, 1)
    #             total += y.size(0)
    #             # Count how many predictions match actual labels
    #             correct += (predicted == y).sum().item()
    #     
    #     test_loss /= len(test_loader)
    #     accuracy = 100 * correct / total
    #     
    #     # Print metrics to monitor training progress
    #     # WHY: Watching these helps detect overfitting (test loss increases while train decreases)
    #     print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}, Test Accuracy = {accuracy:.2f}%")

        # Save model weights after each epoch
        # WHY: Preserves trained model so we can use it later without retraining
        # Currently saves after every epoch - could optimize to save only the best model
        # with open('model_state.pth', 'wb') as f:
        #     save(clf.state_dict(), f)