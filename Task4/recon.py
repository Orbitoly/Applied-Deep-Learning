# a basic CNN classifies CIFAR-10
import torch
import torchvision
import torchvision.transforms as transforms

# 1. data handling (mean, std), images loaded in range [0,1] normalize to [-1, 1]
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((.5, .5, .5), (.5, .5, .5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# take only 4 images of training set:
take_size = 10000
trainset = torch.utils.data.Subset(trainset, range(take_size))

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(trainloader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# 2. define model
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # channels in, channels out, kernel size
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.deconv1 = nn.ConvTranspose2d(16, 6, 6,stride=2)
        self.deconv2 = nn.ConvTranspose2d(6, 3, 6,stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        encoded = self.pool(F.relu(self.conv1(x)))
        encoded = self.pool(F.relu(self.conv2(encoded)))

        reconstructed = self.deconv2(self.deconv1(encoded.view(-1, 16, 5, 5)))
        encoded = torch.flatten(encoded, 1)
        decoded = F.relu(self.fc1(encoded))
        decoded = F.relu(self.fc2(decoded))
        decoded = self.fc3(decoded)
        return decoded, reconstructed


net = Net()

# 3. define loss and optimizer
import torch.optim as optim

reconstruction_criterion = nn.MSELoss()
classification_criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4. train model
PATH = './cifar_net.pth'
run_train = True
if run_train:
    for epoch in range(2):  # loop over dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs, reconstructed = net(inputs)

            classification_loss = classification_criterion(outputs, labels)

            # Reconstruction loss

            reconstruction_loss = reconstruction_criterion(reconstructed, inputs)

            loss = classification_loss + 0.01 * reconstruction_loss  # Adjust the weight for reconstruction loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

# torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# 5. test model
# net = Net()
# net.load_state_dict(torch.load(PATH))
def imshow_reconstructed(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()

outputs, reconstructed  = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))
imshow_reconstructed(torchvision.utils.make_grid(reconstructed))
