# first attempt at a nn module

# import dependencies
import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Get data
train = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
test = datasets.MNIST(root='data', train=False, download=True, transform=ToTensor())
train_loader = DataLoader(train, batch_size=32, shuffle=True)
test_loader = DataLoader(test, batch_size=32, shuffle=False)

# Define Image Classifier and Neural Network
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

# Instantiate model, define loss function and optimizer
clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training loop
if __name__ == "__main__":
    # with open('model_state.pth', 'rb') as f:
    #     clf.load_state_dict(load(f))


    # img = Image.open('img_3.jpg')
    # img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')
    # print(torch.argmax(clf(img_tensor)))


    for epoch in range(10):
        # Training phase
        clf.train()
        train_loss = 0.0
        for batch in train_loader:
            X, y = batch
            X, y = X.to('cuda'), y.to('cuda')
            yhat = clf(X)
            loss = loss_fn(yhat, y)
            
            opt.zero_grad() # Apply backprop
            loss.backward()
            opt.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation/Test phase
        clf.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                X, y = batch
                X, y = X.to('cuda'), y.to('cuda')
                yhat = clf(X)
                loss = loss_fn(yhat, y)
                test_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(yhat, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = 100 * correct / total
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}, Test Accuracy = {accuracy:.2f}%")

    #     with open('model_state.pth', 'wb') as f:
    #         save(clf.state_dict(), f)