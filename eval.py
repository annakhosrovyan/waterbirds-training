import torch
import logging
from tqdm import tqdm  

log = logging.getLogger(__name__)


#       ----------------------------------------------------------------------------------------
#       -------------------Functions for Checking Accuracy and Group Accuracy-------------------
#       ----------------------------------------------------------------------------------------

def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y, _ in tqdm(loader):
            x = x.to(device = device)
            y = y.to(device = device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        accuracy = num_correct / num_samples * 100
        log.info(f"Got {num_correct} / {num_samples} with accuracy {accuracy:.2f}")

    model.train()

    return accuracy


def check_group_accuracy(dataset, model, label, background, dataset_type, device):
    group_correct = 0
    group_total = 0
    for x, y, metadata in tqdm(dataset):
        x = x.to(device = device)
        y = y.to(device = device)

        metadata = metadata.to(device = device)
        _, pred = model(x.unsqueeze(0)).max(1)

        c = metadata
        if dataset_type == "waterbirds":
            c = metadata[0]

        if (y.item() == label) and (c.item() == background): 
            group_total += 1
            if pred == y:
                group_correct += 1
            
    accuracy = group_correct / group_total * 100
    log.info(f"Got {group_correct} / {group_total} with accuracy {accuracy:.2f}")
    
    return accuracy

#       ----------------------------------------------------------------------------------------
#       -------------------------------Accuracy Logging Functions-------------------------------
#       ----------------------------------------------------------------------------------------


def print_accuracy_for_loaders(train_loader, test_loader, model, device):
    log.info("Checking accuracy on Training Set")
    check_accuracy(train_loader, model, device)

    log.info("Checking accuracy on Test Set")
    check_accuracy(test_loader, model, device)


def print_group_accuracies(dataset, model, dataset_type, device):
    log.info("Accuracy on waterbird_water")
    check_group_accuracy(dataset, model, 1, 1, dataset_type, device)

    log.info("Accuracy on waterbird_land")
    check_group_accuracy(dataset, model, 1, 0, dataset_type, device)

    log.info("Accuracy on landbird_land")
    check_group_accuracy(dataset, model, 0, 0, dataset_type, device)

    log.info("Accuracy on landbird_water")
    check_group_accuracy(dataset, model, 0, 1, dataset_type, device)


def performance(train_loader, test_loader, dataset, model, dataset_type, device):
    print_accuracy_for_loaders(train_loader, test_loader, model, device)
    print_group_accuracies(dataset, model, dataset_type, device)


def standard_model_performance(train_loader, test_loader, dataset, model, dataset_type, device):
    log.info("\nEvaluate standard model performance\n")
    performance(train_loader, test_loader, dataset, model, dataset_type, device)
    
            
def algorithm_performance(train_loader, test_loader, dataset, model, dataset_type, device):
    log.info("\nEvaluate Algorithm performance\n")
    performance(train_loader, test_loader, dataset, model, dataset_type, device)