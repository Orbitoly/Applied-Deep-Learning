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
num_epochs = 20
batch_size = 100
learning_rate = 0.001


grid_search_combinations = [100, 0.001, 500]
# for batch_size in batch_sizes_grid_search:
#     for learning_rate in learning_rate_grid_search:
#         for hidden_size in hidden_size_grid_search:
#             grid_search_combinations.append([batch_size, learning_rate, hidden_size])

train_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data/',
                                          train=False,
                                          transform=transforms.ToTensor())
#trim the train dataset to 1000 samples
sizee = 5000
train_dataset, _ = torch.utils.data.random_split(train_dataset, [sizee, len(train_dataset) - sizee])


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


images, labels = next(iter(train_loader))


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
        hidden = out.clone()  # new line to store hidden features
        out = self.fc2(out)
        return out, hidden  # return both the output and hidden features

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train the model
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        outputs,_ = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    print('Epoch [{}/{}]'.format(epoch+1, num_epochs))

#print accuracy:
correct = 0
total = 0
for images, labels in test_loader:
    images = images.reshape(-1, input_size).to(device)
    labels = labels.to(device)
    outputs,_ = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)

    correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
# Get hidden features and original images for all train set
hidden_features = []
original_images = []
labels = []
with torch.no_grad():
    for images, lbls in train_loader:
        images = images.reshape(-1, input_size).to(device)
        lbls = lbls.to(device)
        _, hidden = model(images)
        hidden_features.append(hidden.cpu().numpy())
        original_images.append(images.cpu().numpy())
        labels.append(lbls.cpu().numpy())
hidden_features = np.concatenate(hidden_features, axis=0)
original_images = np.concatenate(original_images, axis=0)
labels = np.concatenate(labels, axis=0)

# Plot t-SNE embeddings of hidden features
tsne = TSNE(n_components=2, random_state=42)
hidden_features_embedded = tsne.fit_transform(hidden_features)

# Plot z_i
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(hidden_features_embedded[:, 0], hidden_features_embedded[:, 1], c=labels)
ax.set_title('t-SNE Embedding of Hidden Features')
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)
plt.show()


# Get hidden features and original images for all train set
hidden_features = []
original_images = []
labels = []
with torch.no_grad():
    for images, lbls in train_loader:
        images = images.reshape(-1, input_size).to(device)
        lbls = lbls.to(device)
        _, hidden = model(images)
        hidden_features.append(hidden.cpu().numpy())
        original_images.append(images.cpu().numpy())
        labels.append(lbls.cpu().numpy())
hidden_features = np.concatenate(hidden_features, axis=0)
original_images = np.concatenate(original_images, axis=0)
labels = np.concatenate(labels, axis=0)

# Plot x_i
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(10, 6),
                       gridspec_kw={'wspace': 0.05, 'hspace': 0.05},
                       sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img = original_images[labels == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='gray', interpolation='nearest')
    ax[i].set_title(str(i))
    ax[i].axis('off')
plt.suptitle('Sample images from the MNIST dataset')
plt.show()
