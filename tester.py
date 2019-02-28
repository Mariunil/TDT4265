import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from dataloaders import load_cifar10
from utils import to_cuda, compute_loss_and_accuracy
import model3
import json


if __name__ == "__main__":
    trainer = model3.Trainer()
    trainer.train()

    file1 = open("model3_test_acc.txt", "w")
    json.dump(trainer.TEST_ACC, file1)
    file1.close()

    file2 = open("model3_val_acc.txt", "w")
    json.dump(trainer.VALIDATION_ACC, file2)
    file2.close()

    file3 = open("model3_train_acc.txt", "w")
    json.dump(trainer.TRAIN_ACC, file3)
    file3.close()

    file4 = open("model3_test_loss.txt", "w")
    json.dump(trainer.TEST_LOSS, file4)
    file4.close()

    file5 = open("model3_train_loss.txt", "w")
    json.dump(trainer.TRAIN_LOSS, file5)
    file5.close()

    file6 = open("model3_val_loss.txt", "w")
    json.dump(trainer.VALIDATION_LOSS, file6)
    file6.close()
