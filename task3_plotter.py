import os
import json
import matplotlib.pyplot as plt



if __name__ == "__main__":
    epoch_list = [0, 0.33, 0.66, 1]

    file1 = open("task3_test_acc.txt", "r")
    task3_test_acc = json.load(file1)
    file1.close()

    file2 = open("task3_train_acc.txt", "r")
    task3_train_acc = json.load(file2)
    file2.close()

    file3 = open("task3_val_acc.txt", "r")
    task3_val_acc = json.load(file3)
    file3.close()

    file4 = open("task3_test_loss.txt", "r")
    task3_test_loss = json.load(file4)
    file4.close()

    file5 = open("task3_train_loss.txt", "r")
    task3_train_loss = json.load(file5)
    file5.close()

    file6 = open("task3_val_loss.txt", "r")
    task3_val_loss = json.load(file6)
    file6.close()

    os.makedirs("plots", exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(12, 8))
    plt.title("Cross Entropy Loss")
    plt.plot(epoch_list, task3_val_loss, label="Validation loss")
    plt.plot(epoch_list, task3_train_loss, label="Training loss")
    plt.plot(epoch_list, task3_test_loss, label="Testing Loss")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_loss.png"))
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.title("Accuracy")
    plt.plot(epoch_list, task3_val_acc, label="Validation Accuracy")
    plt.plot(epoch_list, task3_train_acc, label="Training Accuracy")
    plt.plot(epoch_list, task3_test_acc, label="Testing Accuracy")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_accuracy.png"))
    plt.show()
