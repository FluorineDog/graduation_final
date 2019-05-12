import torch as tf
import torch.nn as nn
import random
import numpy as np


data = [[(float)(random.randrange(-1000, 1000)/1000) for i in range(10)] for j in range(3000)]
labels = [ (int)(1 if (sum(arr) > 0.00001) else 0) for arr in data]

data = tf.Tensor(data)
labels = tf.as_tensor(labels, dtype=tf.int64)
net = nn.Sequential(nn.Linear(10, 2))


criterion = nn.CrossEntropyLoss()
opt = tf.optim.Adam(net.parameters(), lr=0.0001, weight_decay=5e-4)


for iter in range(1000000):
    opt.zero_grad()
    output = net(data)
    loss = criterion(output, labels)
    l = loss.item()
    if iter % 100 == 0:
        # ct = sum([output[i][labels[i]].item() >= output[i][1 - labels[i]].item()  for i in range(3000)])
        ct = sum( [tf.argmax(output[i]).item() == labels[i].item() for i in range(3000)])
        print("loss = ", l, "same = " ,ct)
    loss.backward()
    opt.step()
    pass
