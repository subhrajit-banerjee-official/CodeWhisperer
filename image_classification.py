import torch
import os
from torch import nn
from torchvision import transforms, datasets

# Define the transformation to be applied to the images
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Get the current working directory
current_directory = os.getcwd()

# Construct a relative path
train_folder = "/data/train"
val_folder = "/data/val"

# Join the current directory and relative path to create an absolute path
data_train = current_directory+train_folder
data_val = current_directory +val_folder

#print(f'train_data: {data_train} val_data: {data_val}  current_directory: {current_directory}')

# Load the training and validation data
train_data = datasets.ImageFolder(root=data_train, transform=data_transform)
val_data = datasets.ImageFolder(root=data_val, transform=data_transform)


# Create data loaders for training and validation data
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64)


# Define the convolutional neural network (CNN) architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(28 * 28 * 128, 1024)
        self.fc2 = nn.Linear(1024, 50)

    #The Rectified Linear Unit (ReLU) is an activation function used in neural network. 
    #It’s a non-linear function that outputs the input directly if it is positive; otherwise, it outputs zero.
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 28 * 28 * 128)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the neural network
model = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Train the model
#for epoch in range(10):

for epoch in range(5):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch: {epoch+1} Loss: {running_loss/len(train_loader)}')

# Evaluate the model on validation data
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy: {100 * correct / total}%')
