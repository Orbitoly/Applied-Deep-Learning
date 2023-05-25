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
	img = img / 2 + 0.5 	# unnormalize
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
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = torch.flatten(x, 1)		# flatten all except batch
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

net = Net()


# 3. define loss and optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4. train model
PATH = './cifar_net.pth'
run_train = False
if run_train:
	for epoch in range(2):		# loop over dataset multiple times

		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data

			optimizer.zero_grad()

			outputs = net(inputs)

			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			if i % 2000 == 1999:
				print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
				running_loss = 0.0

	print('Finished Training')

	torch.save(net.state_dict(), PATH)


dataiter = iter(testloader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# 5. test model
net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))
