import logging
from tqdm import tqdm  

log = logging.getLogger(__name__)


def check_group_accuracy(dataset, model, label, background, dataset_type):
    group_correct = 0
    group_total = 0
    for x, y, metadata in tqdm(dataset):
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


def print_group_accuracies(dataset, model, dataset_type):
    log.info("Accuracy on waterbird_water")
    check_group_accuracy(dataset, model, 1, 1, dataset_type)

    log.info("Accuracy on waterbird_land")
    check_group_accuracy(dataset, model, 1, 0, dataset_type)

    log.info("Accuracy on landbird_land")
    check_group_accuracy(dataset, model, 0, 0, dataset_type)

    log.info("Accuracy on landbird_water")
    check_group_accuracy(dataset, model, 0, 1, dataset_type)

