import torch as tf
import torch.nn as nn
import random
import numpy as np


data = [[(float)(random.randrange(-1.0, 1.0)) for i in range(10)] for j in range(30)]
labels = [ (int)(1 if (sum(arr) > 0) else 0) for arr in data]

data = tf.Tensor(data)
labels = tf.as_tensor(labels, dtype=tf.int64)
net = nn.Linear(10, 2)

criterion = nn.CrossEntropyLoss()
opt = tf.optim.SGD(net.parameters(), lr=0.01, momentum=0.0, weight_decay=5e-4)


for iter in range(10000):
    opt.zero_grad()
    output = net(data)
    loss = criterion(output, labels)
    l = loss.item()
    print("loss = ", l)
    loss.backward()
    opt.step()
    pass  
