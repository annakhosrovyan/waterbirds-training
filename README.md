## resnet50_representation

algorithm:
  name: jtt
  params:
    lambda_up: 100
    num_epochs_for_final_model: 300
  data_format:
  - vector
model:
  name: linear_classifier
  linear_config:
    _target_: models.LinearClassifier
  params:
    num_classes: 2
    batch_size: 64
    num_epochs: 60
  data_format:
  - vector
dataset:
  name: resnet50_representation
  in_features: 2048
  paths:
    train_path: C:/Users/User/Desktop/Datasets/resnet50/train.npz
    test_path: C:/Users/User/Desktop/Datasets/resnet50/test.npz
    val_path: C:/Users/User/Desktop/Datasets/resnet50/val.npz
  data_format:
  - vector
optimizer:
  name: adam
  optim:
    _target_: torch.optim.Adam
    lr: 0.01
    weight_decay: 1.0e-05
loss:
  name: cross_entropy
  loss_function:
    _target_: torch.nn.CrossEntropyLoss

Evaluate standard model performance

Checking accuracy on Training Set
Got 4749 / 4795 with accuracy 99.04
Checking accuracy on Test Set
Got 5000 / 5794 with accuracy 86.30
Accuracy on waterbird_water
Got 605 / 642 with accuracy 94.24
Accuracy on waterbird_land
Got 409 / 642 with accuracy 63.71
Accuracy on landbird_land
Got 2236 / 2255 with accuracy 99.16
Accuracy on landbird_water
Got 1750 / 2255 with accuracy 77.61

Validation Group Accuracies

Accuracy on waterbird_water
Got 127 / 133 with accuracy 95.49
Accuracy on waterbird_land
Got 83 / 133 with accuracy 62.41
Accuracy on landbird_land
Got 460 / 467 with accuracy 98.50
Accuracy on landbird_water
Got 325 / 466 with accuracy 69.74

Evaluate JTT performance

Checking accuracy on Training Set
Got 4742 / 4795 with accuracy 98.89
Checking accuracy on Test Set
Got 4943 / 5794 with accuracy 85.31
Accuracy on waterbird_water
Got 612 / 642 with accuracy 95.33
Accuracy on waterbird_land
Got 440 / 642 with accuracy 68.54
Accuracy on landbird_land
Got 2227 / 2255 with accuracy 98.76
Accuracy on landbird_water
Got 1664 / 2255 with accuracy 73.79


## dino_v2_representation

algorithm:
  name: jtt
  params:
    lambda_up: 700
    num_epochs_for_final_model: 300
  data_format:
  - vector
model:
  name: linear_classifier
  linear_config:
    _target_: models.LinearClassifier
  params:
    num_classes: 2
    batch_size: 64
    num_epochs: 60
  data_format:
  - vector
dataset:
  name: dino_v2_representation
  in_features: 768
  paths:
    train_path: C:/Users/User/Desktop/Datasets/dino_v2/train.npz
    test_path: C:/Users/User/Desktop/Datasets/dino_v2/test.npz
    val_path: C:/Users/User/Desktop/Datasets/dino_v2/val.npz
  data_format:
  - vector
optimizer:
  name: sgd
  optim:
    _target_: torch.optim.SGD
    lr: 0.01
    momentum: 0.9
    weight_decay: 1
loss:
  name: cross_entropy
  loss_function:
    _target_: torch.nn.CrossEntropyLoss

Evaluate standard model performance

Checking accuracy on Training Set
Got 4753 / 4795 with accuracy 99.12
Checking accuracy on Test Set
Got 5595 / 5794 with accuracy 96.57
Accuracy on waterbird_water
Got 615 / 642 with accuracy 95.79
Accuracy on waterbird_land
Got 563 / 642 with accuracy 87.69
Accuracy on landbird_land
Got 2253 / 2255 with accuracy 99.91
Accuracy on landbird_water
Got 2164 / 2255 with accuracy 95.96

Validation Group Accuracies

Accuracy on waterbird_water
Got 132 / 133 with accuracy 99.25
Accuracy on waterbird_land
Got 125 / 133 with accuracy 93.98
Accuracy on landbird_land
Got 444 / 467 with accuracy 95.07
Accuracy on landbird_water
Got 365 / 466 with accuracy 78.33

Evaluate JTT performance

Checking accuracy on Training Set
Got 4653 / 4795 with accuracy 97.04
Checking accuracy on Test Set
Got 5228 / 5794 with accuracy 90.23
Accuracy on waterbird_water
Got 639 / 642 with accuracy 99.53
Accuracy on waterbird_land
Got 598 / 642 with accuracy 93.15
Accuracy on landbird_land
Got 2181 / 2255 with accuracy 96.72
Accuracy on landbird_water
Got 1810 / 2255 with accuracy 80.27


## regnet_representation

algorithm:
  name: jtt
  params:
    lambda_up: 20
    num_epochs_for_final_model: 250
  data_format:
  - vector
model:
  name: linear_classifier
  linear_config:
    _target_: models.LinearClassifier
  params:
    num_classes: 2
    batch_size: 64
    num_epochs: 60
  data_format:
  - vector
dataset:
  name: regnet_representation
  in_features: 7392
  paths:
    train_path: C:/Users/User/Desktop/Datasets/swag_pretrained_regnety_128gf_in1k/train.npz
    test_path: C:/Users/User/Desktop/Datasets/swag_pretrained_regnety_128gf_in1k/test.npz
    val_path: C:/Users/User/Desktop/Datasets/swag_pretrained_regnety_128gf_in1k/val.npz
  data_format:
  - vector
optimizer:
  name: adam
  optim:
    _target_: torch.optim.Adam
    lr: 0.001
    weight_decay: 1
loss:
  name: cross_entropy
  loss_function:
    _target_: torch.nn.CrossEntropyLoss

Evaluate standard model performance

Checking accuracy on Training Set
Got 4767 / 4795 with accuracy 99.42
Checking accuracy on Test Set
Got 5475 / 5794 with accuracy 94.49
Accuracy on waterbird_water
Got 621 / 642 with accuracy 96.73
Accuracy on waterbird_land
Got 503 / 642 with accuracy 78.35
Accuracy on landbird_land
Got 2255 / 2255 with accuracy 100.00
Accuracy on landbird_water
Got 2096 / 2255 with accuracy 92.95

Validation Group Accuracies

Accuracy on waterbird_water
Got 129 / 133 with accuracy 96.99
Accuracy on waterbird_land
Got 108 / 133 with accuracy 81.20
Accuracy on landbird_land
Got 467 / 467 with accuracy 100.00
Accuracy on landbird_water
Got 406 / 466 with accuracy 87.12

Evaluate JTT performance

Checking accuracy on Training Set
Got 4773 / 4795 with accuracy 99.54
Checking accuracy on Test Set
Got 5359 / 5794 with accuracy 92.49
Accuracy on waterbird_water
Got 627 / 642 with accuracy 97.66
Accuracy on waterbird_land
Got 539 / 642 with accuracy 83.96
Accuracy on landbird_land
Got 2249 / 2255 with accuracy 99.73
Accuracy on landbird_water
Got 1944 / 2255 with accuracy 86.21