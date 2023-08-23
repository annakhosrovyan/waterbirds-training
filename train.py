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


#       ------------------------------------------------------------------------------------------------------------
#       -----------------------------Error set and dataset upsampling (for 'jtt')-----------------------------
#       ------------------------------------------------------------------------------------------------------------

def construct_error_set(model, data, device): 
    error_set = []
    for index, (x, y, _) in enumerate(tqdm(data)):
        x = x.to(device = device)
        y = y.to(device = device)

        _, pred = model(x.unsqueeze(0)).max(1)
        
        if pred != y:
               error_set.append(index)
    
    return error_set


def construct_upsampled_dataset(model, data, lambda_up, device):
    upsampled_dataset = []

    error_set = construct_error_set(model, data, device)

    for index, (x, y, c) in enumerate(tqdm(data)):
        if index in error_set:
            for i in range(lambda_up):
                upsampled_dataset.append((x, y, c))
        else:
            upsampled_dataset.append((x, y, c))

    return upsampled_dataset