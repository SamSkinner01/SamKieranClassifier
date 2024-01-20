import torch
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image

# Get the data
train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

dataset = DataLoader(train_data, batch_size=64, shuffle=True)

# Define the model
class Model(nn.Module):
    def __init__(self):
        super().__init__() # inits the nn.Module class

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64,64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (28-6) * (28-6), 10)
        )

    def forward(self, x):
        return self.model(x)
    
if __name__ == '__main__':
    model = Model().to('cuda')
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        for batch in dataset:
            X, y = batch
            X, y = X.to('cuda'), y.to('cuda')
            prediction = model(X)
            loss = loss_fn(prediction, y) # y is true value

            # Backpropagation
            optimizer.zero_grad() # reset the gradients
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1} done with loss {loss.item()}')

    # Save the model
    with open('mnist.pth', 'wb') as f:
        save(model.state_dict(), f)


    # test the model
    model.eval()
    test_dataset = DataLoader(test_data, batch_size=64, shuffle=True)
    correct = 0
    total = 0
    for batch in test_dataset:
        X, y = batch
        X, y = X.to('cuda'), y.to('cuda')
        prediction = model(X)
        _, prediction = torch.max(prediction, dim=1)
        correct += torch.sum(prediction == y)
        total += len(y)
    print(f'Accuracy: {correct/total}')
    
