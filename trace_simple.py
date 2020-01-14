from trace import convolve2d, cos, load_trace, k_cluster, delta_angles
from calibrate_gravity import get_gravity, remove_gravity
import numpy as np

difference_filter = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

vectors = []
labels = []

# Load the simple gestures
for i in range(1, 6):
    for dir in ['left', 'right', 'up', 'down']:
        trace = load_trace("data/simple/%s%d.csv" % (dir, i))

        # Calculate and remove the force of gravity
        gravity = get_gravity(load_trace("data/simple/rest.csv"))
        trace = remove_gravity(trace, gravity)
        #left_trace = convolve2d(left_trace, difference_filter)

        # Find the first acceleration cluster
        clusters = k_cluster(trace)

        cluster = clusters[0][0]
        cluster_angles = [cos(cluster, [1.0, 1.0, 0]), cos(cluster, [1.0, 0.0, 1.0])]

        # Append it to the list
        vectors.append(cluster_angles)
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

labels_enc_1hot = torch.LongTensor([int_to_one_hot(label) for label in labels_enc])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 5)
        self.fc2 = nn.Linear(5, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
svm_crite = nn.MultiLabelMarginLoss()
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
    if epoch % 100 == 0:
        print(running_loss)

"""for epoch in range(200):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(torch.Tensor(vectors))
    loss = svm_crite(outputs, labels_enc_1hot)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss = loss.item()
    if epoch % 1 == 0:
        print(running_loss)"""

correct = 0
total = 0
for i in range(len(vectors)):
    x = vectors[i]
    y = labels[i]

    outputs = net(torch.Tensor(x))
    idx_max = torch.argmax(outputs).item()
    if y == label_dec[idx_max]:
        correct += 1
    total += 1
    #print("%s == %s" % (y, label_dec[idx_max]))

print("Training accuracy: %d percent" % ((correct / total) * 100))
