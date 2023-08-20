import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms 

from data_processing import get_train_data


def show_image(train_data):
    train_data = get_train_data()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    figure = plt.figure(figsize = (5, 5))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_data), size = (1,)).item()
        img, label, metadata = train_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title("Waterbird") if label.item() == 1 else  plt.title("Landbird")
        plt.axis("off")
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()