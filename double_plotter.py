import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from dataloaders import load_cifar10
from utils import to_cuda, compute_loss_and_accuracy
import task3.py



if __name__ == "__main__":
    trainer = task3.Trainer()
    trainer.train()

    file1 = open("resnet_test_acc.txt, w")
    file1.write(task3.Trainer.TEST_ACC)

    file2 = open("resnet_val_acc.txt, w")
    file2.write(task3.Trainer.VALIDATION_ACC)

    file3 = open("resnet_train_acc.txt, w")
    file3.write(task3.Trainer.TRAIN_ACC)

    file4 = open("resnet_test_loss.txt, w")
    file4.write(task3.Trainer.TEST_LOSS)

    file5 = open("resnet_train_loss.txt, w")
    file5.write(task3.Trainer.TRAIN_LOSS)

    file6 = open("resnet_val_loss.txt, w")
    file6.write(task3.Trainer.VALDIATION_LOSS)
