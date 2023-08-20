import torch
import logging
from tqdm import tqdm  

log = logging.getLogger(__name__)

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

        log.info(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")

    model.train()


def check_waterbirds_images_group_accuracy(data, model, label, background, device):
    group_correct = 0
    group_total = 0
    for x, y, metadata in tqdm(data):
        x = x.to(device = device)
        y = y.to(device = device)

        metadata = metadata.to(device = device)
        _, pred = model(x.unsqueeze(0)).max(1)

        if (y.item() == label) and (metadata[0].item() == background): 
            group_total += 1
            if pred == y:
                group_correct += 1
    
    log.info(f"Got {group_correct} / {group_total} with accuracy {float(group_correct)/float(group_total)*100:.2f}")


def check_resnet50_representation_group_accuracy(data, model, label, background, device):
    group_correct = 0
    group_total = 0
    for x, y, c in tqdm(data):
        x = x.to(device = device)
        y = y.to(device = device)
        c = c.to(device = device)
        _, pred = model(x.unsqueeze(0)).max(1)
        if (y.item() == label) and (c.item() == background): 
            group_total += 1
            if pred == y:
                group_correct += 1
    
    log.info(f"Got {group_correct} / {group_total} with accuracy {float(group_correct)/float(group_total)*100:.2f}")



#       ###########################################
#       ###########################################
#           Accuracy Logging Functions
#       ###########################################
#       ###########################################


def print_accuracy_for_loaders(train_loader, test_loader, model, device):
      log.info("Checking accuracy on Training Set")
      check_accuracy(train_loader, model, device)

      log.info("Checking accuracy on Test Set")
      check_accuracy(test_loader, model, device)


def print_group_accuracies_for_resnet50_representation(test_data, model, device):
      log.info("Accuracy on waterbird_water")
      check_resnet50_representation_group_accuracy(test_data, model, 1, 1, device)

      log.info("Accuracy on waterbird_land")
      check_resnet50_representation_group_accuracy(test_data, model, 1, 0, device)

      log.info("Accuracy on landbird_land")
      check_resnet50_representation_group_accuracy(test_data, model, 0, 0, device)

      log.info("Accuracy on landbird_water")
      check_resnet50_representation_group_accuracy(test_data, model, 0, 1, device)


def print_group_accuracies_for_waterbirds_images(test_data, model, device):
      log.info("Accuracy on waterbird_water")
      check_waterbirds_images_group_accuracy(test_data, model, 1, 1, device)

      log.info("Accuracy on waterbird_land")
      check_waterbirds_images_group_accuracy(test_data, model, 1, 0, device)

      log.info("Accuracy on landbird_land")
      check_waterbirds_images_group_accuracy(test_data, model, 0, 0, device)

      log.info("Accuracy on landbird_water")
      check_waterbirds_images_group_accuracy(test_data, model, 0, 1, device)

