import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn, save, load
from PIL import Image

NUM_EPOCHS = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get MNIST Dataset
train = datasets.MNIST(root="data", download=True, train=True, transform=transforms.ToTensor(), )
dataset = DataLoader(train, 32)


# Image Classifier Neural Network
class ImageClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Conv2d(1, 32, (3, 3)),
        nn.ReLU(),
        nn.Conv2d(32, 64, (3, 3)),
        nn.ReLU(),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64*(28-6)*(28-6), 10)
    )

  def forward(self, x):
    return self.model(x)
  
# Net object
net = ImageClassifier().to(DEVICE)

# Optimizer object
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_function = nn.CrossEntropyLoss()

if __name__ == "__main__":
  # Training
  for epoch in range(NUM_EPOCHS):
    for batch in dataset:
      x, y = batch
      x, y = x.to(DEVICE), y.to(DEVICE)

      # Forward
      yhat = net(x)
      loss = loss_function(yhat, y)

      # Backward
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
    print(f"Epoch: {epoch} Loss: {loss.item()}")

    with open('model_state.pt', 'wb') as f:
      save(net.state_dict(), f)

    # Prediction
    with open('model_state.pt', 'rb') as f:
        net.load_state_dict(load(f))

    img = Image.open('img_1.jpg')
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
    print(torch.argmax(net(img_tensor)))
