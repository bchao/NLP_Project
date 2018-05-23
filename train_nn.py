# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py#L37-L49

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pickle
import numpy as np
import time


start = time.asctime()
print(start)

root = '/Users/Daniel/Documents/University of Chicago/TTIC 31190/project2.7/'

train_names = ['ROCStories__spring2016 - ROCStories_spring2016',
               'ROCStories_winter2017 - ROCStories_winter2017']
val_name = 'cloze_test_val__spring2016 - cloze_test_ALL_val'
test_name = 'cloze_test_test__spring2016 - cloze_test_ALL_test'

# train1 = pickle.load(open(root + 'embeddings_' + train_names[0] + '.pkl', 'rb'), encoding='latin1')
# train2 = pickle.load(open(root + 'embeddings_' + train_names[1] + '.pkl', 'rb'), encoding='latin1')
val = pickle.load(open(root + 'embeddings_' + val_name + '.pkl', 'rb'), encoding='latin1')
val_labels = pickle.load(open(root + 'labels_' + val_name + '.pkl', 'rb'), encoding='latin1')
test = pickle.load(open(root + 'embeddings_' + test_name + '.pkl', 'rb'), encoding='latin1')
test_labels = pickle.load(open(root + 'labels_' + test_name + '.pkl', 'rb'), encoding='latin1')


def transform_val_test_LS(data, labels):
    out = []
    for i in range(len(data)):
        out.append([data[i][3] + data[i][4], int(int(labels[i]) == 1)])  # s4 + s5.1 with 0/1 label
        out.append([data[i][3] + data[i][5], int(int(labels[i]) == 2)])  # s4 + s5.2 with 0/1 label
    return out


def transform_val_test_NC(data, labels):
    out = []
    for i in range(len(data)):
        out.append([data[i][4], int(int(labels[i]) == 1)])  # s5.1 with 0/1 label
        out.append([data[i][5], int(int(labels[i]) == 2)])  # s5.2 with 0/1 label
    return out


val_load = transform_val_test_LS(val, val_labels)
test_load = transform_val_test_LS(test, test_labels)
# val_load = transform_val_test_NC(val, val_labels)
# test_load = transform_val_test_NC(test, test_labels)


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 4800
hidden_size = [256, 64]
num_classes = 2
num_epochs = 10
learning_rate = 0.01
batch_size = 32  # must be an even number!!

train_dataset = val_load
test_dataset = test_load

# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Neural network
# hidden_size is a list. The length of hidden_size is the number of hidden layers
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList()

        if isinstance(hidden_size, int):
            # create 1-element list if only one hidden layer
            hidden_size = [hidden_size]

        self.layers.append(nn.Linear(input_size, hidden_size[0]))
        num_layers = len(hidden_size)
        for i in range(1, num_layers):
            self.layers.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
        self.layers.append(nn.Linear(hidden_size[num_layers-1], num_classes))

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layers[0](x)
        for i in range(1, len(self.layers)):
            out = self.relu(out)
            out = self.layers[i](out)

        return out

model = NeuralNet(input_size, hidden_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    print(time.asctime())
    for i, (sentences, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        # sentences = sentences.reshape(-1, 28 * 28).to(device)
        sentences = sentences.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(sentences)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for sentences, labels in test_loader:
        # sentences = sentences.reshape(-1, 28 * 28).to(device)
        sentences = sentences.to(device)
        labels = labels.to(device)
        outputs = model(sentences)

        true_labels = []
        true_preds = []
        for i in range(0, len(labels), 2):
            if labels[i] == 1:
                # first option is correct
                true_labels.append(1)
            else:
                # second option is correct
                true_labels.append(2)
            if outputs[i][1] > outputs[i][0]:
                # first option has higher softmax score
                true_preds.append(1)
            else:
                true_preds.append(2)
        true_labels = torch.Tensor(true_labels)
        true_preds = torch.Tensor(true_preds)

        # _, predicted = torch.max(outputs.data, 1)
        # total += labels.size(0)
        total += labels.size(0) / 2  # divide by 2 b/c we have two examples per story
        # correct += (predicted == labels).sum().item()
        correct += (true_preds == true_labels).sum().item()

    print('Accuracy of the network on the test set: {} %'.format(100 * correct / total))


end = time.asctime()

print("\nstart: ", start)
print("end: ", end)

# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
