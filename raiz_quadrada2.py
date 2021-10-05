# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 09:04:49 2021

@author: Frederick Moschkowich
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

v = torch.range(1,10).view(10,1)
l = v**2

model = nn.Sequential(nn.Linear(1,10), 
                      nn.ReLU(), 
                      nn.Linear(10, 1))

criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.001)

epochs = 1000
for e in range(epochs):
        output = model(v)
        loss = criterion(output, 1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
print(f"Training Loss: {loss.item()}")
        
predicted = model(v).detach().numpy()
plt.plot(v.detach(),l.detach(), 'ro')
plt.plot(v.detach(),predicted, 'b')
plt.show()
