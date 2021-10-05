# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 08:50:23 2021

@author: Frederick Moschkowich
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

X = torch.range(1,10).view(10,1)
y = X**2

n_sample, n_features = X.shape
model = nn.Linear(n_features, n_features)

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 200

for e in range(epochs):
        #forward and loss
        y_predict = model(X)
        loss = criterion(y_predict,y)
        #backward
        loss.backward()
        #weights update and zero grad
        optimizer.step()
        optimizer.zero_grad()
        if (e+1)%10 == 0: print(f'epoch: {e+1} loss: {loss.item():.4f}')
        
predicted = model(X).detach().numpy()
plt.plot(X,y, 'ro')
plt.plot(X,predicted, 'b')
plt.show()
