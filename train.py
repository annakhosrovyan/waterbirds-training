import torch
from tqdm import tqdm  


def train_model(model, num_epochs, train_loader, criterion, optimizer, device):
    for epoch in range(num_epochs):
        for batch_idx, (data, targets, _) in enumerate(tqdm(train_loader)):
            data = data.to(device = device)
            targets = targets.to(device = device)

            scores = model(data)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()
  
            optimizer.step()