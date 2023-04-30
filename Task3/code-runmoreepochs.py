# based on https://github.com/milindmalshe/Fully-Connected-Neural-Network-PyTorch/blob/master/FCN_MNIST_Classification_PyTorch.py
import torch
import torch.nn as nn
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
num_epochs = 10
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data/',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
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
acc_train_errors=[]
acc_test_errors=[]

seeds = [1,3,5,15,169]
seeds_test_errors = []
for seed in seeds:
    acc_test_errors = []
    torch.manual_seed(seed)

    for epoch in range(num_epochs):
        errors = 0
        wrong = 0
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

        print('Epoch [{}/{}], Average Loss: {:.4f}'.format(epoch+1, num_epochs, errors/total))
        acc_train_errors.append(errors/total)

        test_errors = 0
        wrong = 0
        total = 0

        for i, (images, labels) in enumerate(test_loader):
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            test_loss = criterion(outputs, labels)
            test_errors += test_loss.item()


        print('Test Average Loss: {:.4f}'.format(test_errors / total))
        acc_test_errors.append(test_errors/total)

    print("Seed finished: ", seed)
    seeds_test_errors.append(acc_test_errors)
    print("seeds_test_errors errors: ", seeds_test_errors)

#seeds_test_errors is a list of lists, each list contains the test errors for a seed
#plot seeds_test_errors
for i in range(len(seeds_test_errors)):
    plt.plot(range(1, num_epochs+1), seeds_test_errors[i], label='Seed {}'.format(seeds[i]))

#add plot of average test error with dotted line
avg_test_errors = []
for i in range(num_epochs):
    avg_test_errors.append(sum([seeds_test_errors[j][i] for j in range(len(seeds_test_errors))])/len(seeds_test_errors))
plt.plot(range(1, num_epochs+1), avg_test_errors, label='Average Test Error', linestyle='dashed')
print("Mean test error for last epoch: ", avg_test_errors[-1])
print("Standard deviation for last epoch: ", np.std([seeds_test_errors[j][-1] for j in range(len(seeds_test_errors))]))

# plt.plot(range(1, num_epochs+1), acc_train_errors, label='Training Loss')
# plt.plot(range(1, num_epochs+1), acc_test_errors, label='Test Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()
plt.show()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
    missclassified = []
    for i, (images, labels) in enumerate(test_loader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        for j in range(len(labels)):
            if predicted[j] != labels[j]:
                missclassified.append((images[j], predicted[j], labels[j]))
                break
        if len(missclassified) == 10:
            break

    missclassified_tensors = [t[0] for t in missclassified]
    missclassified_pred = [t[1] for t in missclassified]
    missclassified_labels = [t[2] for t in missclassified]

    missclassified = torch.stack(missclassified_tensors)
    missclassified = missclassified.reshape(-1, 28, 28)
    fig, axs = plt.subplots(2, 3)
    for ii in range(2):
        for jj in range(3):
            idx = 3 * ii + jj
            axs[ii, jj].imshow(missclassified[idx].squeeze())

            axs[ii, jj].set_title("Predicted:" + str(missclassified_pred[idx].item()) + " Actual:" + str(
                missclassified_labels[idx].item()))
            axs[ii, jj].axis('off')
            fig.subplots_adjust(hspace=0.4)

    plt.show()

# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
