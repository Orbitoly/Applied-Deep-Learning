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
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train the model
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    print('Epoch [{}/{}]'.format(epoch+1, num_epochs))


# Let zi = σ(W (1)T xi + b(1)) represent the hidden features of the image xi, obtained from applying the first layer on the input. Attach to your report a plot with the 2D embedding of zi using tSNE, for all zi in the train set. Each 2D point should be colored based on its label:
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Get the hidden features of the train set
hidden_features = []
saved_labels = []
for i, (images, labels) in enumerate(train_loader):
    images = images.reshape(-1, input_size).to(device)
    labels = labels.to(device)
    saved_labels.append(labels)
    outputs = model(images)
    hidden_features.append(outputs.detach().numpy())

hidden_features = np.concatenate(hidden_features, axis=0)

# Apply tSNE
tsne = TSNE(n_components=2, random_state=0)
print("before tsne fit transform")

X_2d = tsne.fit_transform(hidden_features)

print("after tsne fit transform")
print(X_2d)

# Create a list of colors based on the labels
labels = saved_labels
palette = sns.color_palette('bright', len(set(labels)))
colors = [palette[label] for label in labels]


plt.figure(figsize=(8, 8))
#scatter with colors:
sns.scatterplot(X_2d[:, 0], X_2d[:, 1], hue=labels, palette=palette, legend='full')
# plt.scatter(X_2d[:, 0], X_2d[:, 1], s=10, c=colors)

plt.title('tSNE Visualization')
plt.xlabel('tSNE Component 1')
plt.ylabel('tSNE Component 2')


plt.show()


# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
