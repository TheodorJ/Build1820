from trace import convolve2d, cos, load_trace, k_cluster, delta_angles
from calibrate_gravity import get_gravity, remove_gravity
import numpy as np

difference_filter = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

vectors = []
labels = []

num_points = {'shivani': 20, 'tj': 40}

# Load the simple gestures
for name in ['shivani', 'tj']:
    for i in range(1, int(num_points[name]/4) + 1):
        for dir in ['left', 'right', 'up', 'down']:
            print("data/simple/%s/%s%d.csv" % (name, dir, i))
            trace = load_trace("data/simple/%s/%s%d.csv" % (name, dir, i))

            # Calculate and remove the force of gravity
            #gravity = get_gravity(load_trace("data/simple/rest.csv"))
            #trace = remove_gravity(trace, gravity)
            trace = convolve2d(trace, difference_filter)

            # Find the first acceleration cluster
            clusters = k_cluster(trace)

            cluster = clusters[0][0]
            cluster_angles = [cos(cluster, [1.0, 1.0, 0]), cos(cluster, [1.0, 0.0, 1.0])]

            # Append it to the list
            vectors.append(cluster)
            labels.append(dir)

# Now let's try training a basic SVM for to classify these
label_enc = {'left': 0, 'right': 1, 'up': 2, 'down': 3}
label_dec = {0: 'left', 1: 'right', 2: 'up', 3: 'down'}

import torch
import torch.nn as nn
import torch.nn.functional as F

labels_enc = [label_enc[label] for label in labels]
def int_to_one_hot(x):
    y = [0.0, 0.0, 0.0, 0.0]
    y[x] = 1.0
    return y

labels_enc_1hot = torch.FloatTensor([int_to_one_hot(label) for label in labels_enc])


# Separate train and validation data
indices = list(np.random.permutation(60))
val_indices = indices[:15]
train_indices = indices[15:]

val_vectors = [vectors[i] for i in val_indices]
vectors = [vectors[i] for i in train_indices]

val_labels = [labels[i] for i in val_indices]
labels = [labels[i] for i in train_indices]

val_labels_enc = [labels_enc[i] for i in val_indices]
labels_enc = [labels_enc[i] for i in train_indices]

val_labels_enc_1hot = [labels_enc_1hot[i] for i in val_indices]
labels_enc_1hot = labels_enc_1hot[train_indices]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 8)
        self.fc3 = nn.Linear(8, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x

net = Net()


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
svm_crite = nn.MultiLabelSoftMarginLoss()
optimizer = optim.Adam(net.parameters())

for epoch in range(8000):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(torch.Tensor(vectors))
    loss = criterion(outputs, torch.LongTensor(labels_enc))
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss = loss.item()
    if epoch % 50 == 0:
        print(running_loss)

        correct = 0
        total = 0
        for i in range(len(val_vectors)):
            x = val_vectors[i]
            y = val_labels_enc[i]

            outputs = net(torch.Tensor(x))
            idx_max = torch.argmax(outputs).item()
            if y == idx_max:
                correct += 1
            total += 1
            #print("%s == %s" % (y, label_dec[idx_max]))

        print("Validation accuracy: %d percent" % ((correct / total) * 100))

"""# Retrain as an SVM
for epoch in range(200):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(torch.Tensor(vectors))
    loss = svm_crite(outputs, labels_enc_1hot)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss = loss.item()
    if epoch % 10 == 0:
        print(running_loss)

        correct = 0
        total = 0
        for i in range(len(val_vectors)):
            x = val_vectors[i]
            y = val_labels_enc[i]

            outputs = net(torch.Tensor(x))
            idx_max = torch.argmax(outputs).item()
            if y == idx_max:
                correct += 1
            total += 1
            #print("%s == %s" % (y, label_dec[idx_max]))

        print("Validation accuracy: %d percent" % ((correct / total) * 100))"""

correct = 0
total = 0
for i in range(len(vectors)):
    x = vectors[i]
    y = labels_enc[i]

    outputs = net(torch.Tensor(x))
    idx_max = torch.argmax(outputs).item()
    if y == idx_max:
        correct += 1
    total += 1
    #print("%s == %s" % (y, label_dec[idx_max]))

print("Training accuracy: %d percent" % ((correct / total) * 100))

correct = 0
total = 0
for i in range(len(val_vectors)):
    x = val_vectors[i]
    y = val_labels_enc[i]

    outputs = net(torch.Tensor(x))
    idx_max = torch.argmax(outputs).item()
    if y == idx_max:
        correct += 1
    total += 1
    #print("%s == %s" % (y, label_dec[idx_max]))

print("Validation accuracy: %d percent" % ((correct / total) * 100))
