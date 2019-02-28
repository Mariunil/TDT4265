import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from dataloaders import load_cifar10
from utils import to_cuda, compute_loss_and_accuracy
import model3.py


if __name__ == "__main__":
    trainer = task3.Trainer()
    trainer.train()

    file1 = open("model3_test_acc.txt, w")
    file1.write(model3.Trainer.TEST_ACC)

    file2 = open("model3_val_acc.txt, w")
    file2.write(model3.Trainer.VALIDATION_ACC)

    file3 = open("model3_train_acc.txt, w")
    file3.write(model3.Trainer.TRAIN_ACC)

    file4 = open("model3_test_loss.txt, w")
    file4.write(model3.Trainer.TEST_LOSS)

    file5 = open("model3_train_loss.txt, w")
    file5.write(model3.Trainer.TRAIN_LOSS)

    file6 = open("model3_val_loss.txt, w")
    file6.write(model3.Trainer.VALDIATION_LOSS)