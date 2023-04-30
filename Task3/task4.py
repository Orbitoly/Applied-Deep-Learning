# based on https://github.com/milindmalshe/Fully-Connected-Neural-Network-PyTorch/blob/master/FCN_MNIST_Classification_PyTorch.py
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

batch_sizes_grid_search = [3, 20, 100, 350, 400]
learning_rate_grid_search = [0.001, 0.01, 0.1]
hidden_size_grid_search = [10, 100, 500, 1000]

grid_search_combinations = []
for batch_size in batch_sizes_grid_search:
    for learning_rate in learning_rate_grid_search:
        for hidden_size in hidden_size_grid_search:
            grid_search_combinations.append([batch_size, learning_rate, hidden_size])

validation_size = 10000
# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data/',
                                          train=False,
                                          transform=transforms.ToTensor())

# Split train dataset into train / validation
train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - validation_size, validation_size])


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


images, labels = next(iter(train_loader))


fig, axs = plt.subplots(2, 5)
for ii in range(2):
    for jj in range(5):
        idx = 5 * ii + jj
        axs[ii, jj].imshow(images[idx].squeeze())
        axs[ii, jj].set_title(labels[idx].item())
        axs[ii, jj].axis('off')
plt.show()


# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train the model
total_step = len(train_loader)

params_test_errors = []
params_validation_errors = []
for params in grid_search_combinations:
    acc_test_errors = []
    acc_validation_errors = []

    for epoch in range(num_epochs):
        errors = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            total += labels.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            errors += loss.item()

        validation_errors = 0
        total = 0

        for i, (images, labels) in enumerate(validation_loader):
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            validation_loss = criterion(outputs, labels)
            validation_errors += validation_loss.item()


        #print('Epoch [{}/{}], Validation Average Loss: {:.4f}'.format(epoch+1, num_epochs, validation_errors/total))
        acc_validation_errors.append(validation_errors/total)

        test_errors = 0
        total = 0

        for i, (images, labels) in enumerate(test_loader):
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            test_loss = criterion(outputs, labels)
            test_errors += test_loss.item()


        #print('Test Average Loss: {:.4f}'.format(test_errors / total))
        acc_test_errors.append(test_errors/total)

    params_test_errors.append(acc_test_errors)
    params_validation_errors.append(acc_validation_errors)

    print("Params finished: ", params)
    print("Minimum validation error for params: ", min(acc_validation_errors))
    print("Test error for minimum validation error: ", acc_test_errors[acc_validation_errors.index(min(acc_validation_errors))])





# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
