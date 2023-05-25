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


take_size = 1000
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


def imshow(img, title=None):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


dataiter = iter(trainloader)
images, labels = next(dataiter)


test_dataiter = iter(testloader)
test_images, test_labels = next(dataiter)

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
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.deconv1 = nn.ConvTranspose2d(16, 6, 5)
        self.deconv2 = nn.ConvTranspose2d(6, 3, 5)
        self.unpool = nn.MaxUnpool2d(2, 2)


    def forward(self, x):
        x,indicies1 = self.pool(F.relu(self.conv1(x)))
        x_layer1 = x.clone()
        x,indicies2 = self.pool(F.relu(self.conv2(x)))
        x_layer2 = x.clone()
        net.indicies1 = indicies1
        net.indicies2 = indicies2

        #create recon tensor holding the reconstructions for each layer:
        recon = self.deconv1(F.relu(self.unpool(x,indicies2)))
        recon = self.deconv2(F.relu(self.unpool(recon,indicies1)))

        x = torch.flatten(x, 1)  # flatten all except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, recon , x_layer1, x_layer2


net = Net()

# 3. define loss and optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
recon_criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4. train model
PATH = './cifar_net.pth'
run_train = False
l = 1
epochs = 40
test_accuracy=[]
if run_train:
    for epoch in range(epochs):  # loop over dataset multiple times
        if epoch % 5 == 0:
            print(f'epoch {epoch + 1} / {epochs}')
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs,recons,_,_ = net(inputs)
            cross_loss = criterion(outputs, labels)
            recon_loss = recon_criterion(recons, inputs)
            loss = cross_loss + l * recon_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        correct = 0
        total = 0
        with torch.no_grad():
            for test_d in testloader:
                test_img, test_lbl = test_d
                outputs, recons,_,_ = net(test_img)
                _, predicted = torch.max(outputs.data, 1)
                total += test_lbl.size(0)
                correct += (predicted == test_lbl).sum().item()


        test_accuracy.append(100 * correct / total)
    print('Finished Training')

    torch.save(net.state_dict(), PATH)

else:
    net = Net()
    net.load_state_dict(torch.load(PATH))
#plot test accuracy over epochs:
plt.plot(test_accuracy)
plt.ylabel('test accuracy')
plt.xlabel('epoch')
plt.show()


imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# 5. test model
# net = Net()
# net.load_state_dict(torch.load(PATH))

outputs,recons,_,_ = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

# show reconstructed images:
imshow(torchvision.utils.make_grid(recons.detach()))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs,recons,_,_ = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')



train_img, train_lbl = trainset[0]  # Select an image from the train set
test_img, test_lbl = testset[3]  # Select an image from the test set
def generate_reconstructions(image, layer):
    _, reconstructions,layer1,layer2 = net(image.unsqueeze(0))

    print(layer1.shape) #size: [1, 6, 14, 14]
    print(layer2.shape) #size: [1, 16, 5, 5]
    #go over 6 channels of layer 1, zero out all but one channel, use deconv1 and deconv2 to reconstruct image:
    #create subplot for layer 1 with 2 rows and 3 columns:
    layer1_channels_num = 6
    for channel in range(layer1_channels_num):
        layer1_chanel_i = layer1.clone()
        layer1_chanel_i[0][channel+1:] = 0
        layer1_chanel_i[0][:channel] = 0
        recon_image = net.deconv2(F.relu(net.unpool(layer1_chanel_i, net.indicies1)))

        imshow(torchvision.utils.make_grid(recon_image.detach()), title=f'layer1 channel {channel}')

    layer2_channels_num = 16
    layer2_num_channels_to_show = 3
    random_channels_layer2 = np.random.choice(layer2_channels_num, layer2_num_channels_to_show, replace=False)
    for channel in random_channels_layer2:
        layer2_chanel_i = layer2.clone()
        layer2_chanel_i[0][channel+1:] = 0
        layer2_chanel_i[0][:channel] = 0
        recon_image = net.deconv1(F.relu(net.unpool(layer2_chanel_i, net.indicies2)))
        recon_image = net.deconv2(F.relu(net.unpool(recon_image, net.indicies1)))
        imshow(torchvision.utils.make_grid(recon_image.detach()), title=f'layer2 channel {channel}')

imshow(torchvision.utils.make_grid(test_img.detach()))

train_reconstructions = generate_reconstructions(test_img, 0)  # Generate reconstructions for the first convolutional layer
#test_reconstructions = generate_reconstructions(test_img, 1)  # Generate reconstructions for the second convolutional layer

# for recon_image in train_reconstructions:
#     imshow(torchvision.utils.make_grid(recon_image.detach()))