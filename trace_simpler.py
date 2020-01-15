from trace import load_trace, cos
from calibrate_gravity import get_gravity, remove_gravity
import numpy as np

vectors = []
labels = []

num_points = {'shivani': 20, 'tj': 40}
gravity = None

# Load the simple gestures
for name in ['shivani', 'tj']:
    for i in range(1, int(num_points[name]/4) + 1):
        for dir in ['left', 'right', 'up', 'down']:
            print("data/simple/%s/%s%d.csv" % (name, dir, i))
            trace = load_trace("data/simple/%s/%s%d.csv" % (name, dir, i))

            # Calculate and remove the force of gravity
            gravity = trace[0] #get_gravity(load_trace("data/simple/rest.csv"))
            trace = remove_gravity(trace, gravity)

            # Append it to the list
            vectors.append(trace)
            labels.append(dir)

gravity2 = (gravity[1], gravity[2])

label_enc = {'left': 0, 'right': 1, 'up': 2, 'down': 3}
label_dec = {0: 'left', 1: 'right', 2: 'up', 3: 'down'}
labels_enc = [label_enc[label] for label in labels]

correct = 0
total = 0
for i in range(len(vectors)):
    x = vectors[i]
    y = labels_enc[i]

    pred = None

    # h_total = 0.
    # v_total = 0.
    cos_vec = None
    cross = None
    maxp = None
    maxa = None
    for p in x:
        a = p[1]**2 + p[2]**2
        if a > 2.5**2:
            cos_vec = cos(gravity2, (p[1], p[2]))
            cross = gravity[1] * p[2] - gravity[2] * p[1]
            if cos_vec > 0.707:
                pred = 2
            elif cos_vec < -0.707:
                pred = 3
            else:
                if cross > 0:
                    pred = 1
                else:
                     pred = 0
            break
        else:
            if maxa is None or a > maxa:
                maxa = a
                maxp = p

    if pred is None:
        cos_vec = cos(gravity2, (maxp[1], maxp[2]))
        cross = gravity[1] * maxp[2] - gravity[2] * maxp[1]
        if cos_vec > 0.707:
            pred = 3
        elif cos_vec < -0.707:
            pred = 2
        else:
            if cross > 0:
                pred = 1
            else:
                pred = 0

    # if (h_total > v_total):
    #     if (y < 2):
    #         correct += 1
    # else:
    #     if (y >= 2):
    #         correct += 1

    # outputs = net(torch.Tensor(x))
    # idx_max = torch.argmax(outputs).item()
    if y == pred:
        correct += 1
    else:
        print(i)
        print(cos_vec)
        print(cross)
        print(label_dec[pred])
        print(label_dec[y])
    total += 1
    #print("%s == %s" % (y, label_dec[idx_max]))

print("Validation accuracy: %d percent" % ((correct / total) * 100))
