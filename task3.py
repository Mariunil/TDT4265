import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from dataloaders import load_cifar10
from utils import to_cuda, compute_loss_and_accuracy

# github: https://github.com/Mariunil/TDT4265.git

class Model( nn.Module ):
    
    def __init__( self, image_channels, num_classes ):
        super().__init__()
        self.model = torchvision.models.resnet18( pretrained = True )
        self.model.fc = nn.Linear( 512*4 , 10 )      # No need to apply softmax ,
                                                      # as this is done in nn.CrossEntropyloss

        for param in self.model.parameters():        # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters():     # Unfreeze the last fully - connected
            param.requires_grad = True               # layer
        for param in self.model.layer4.parameters(): # Unfreeze the last 5 c on vo lu ti on al
            param.requires_grad = True               # layers


    def forward(self, x):
        x = nn.functional.interpolate(x , scale_factor =8)
        x = self.model(x)
        return x


class Trainer:
    def __init__(self):
        """
        Initialize our trainer class.
        Set hyperparameters, architecture, tracking variables etc.
        """
         # Define hyperparameters
        self.epochs = 5
        self.batch_size = 32
        self.learning_rate = 0.0005
        self.momentum = 0
        self.L2 = 0
        self.nesterov = False
        self.early_stop_count = 4
        self.should_anneal = False
        self.T = 5
        self.t = 0
        self.a0 = 5e-2

        # Architecture

        # Since we are doing multi-class classification, we use the CrossEntropyLoss
        self.loss_criterion = nn.CrossEntropyLoss()
        # Initialize the mode
        self.model = Model(image_channels=3, num_classes=10)
        # Transfer model to GPU VRAM, if possible.
        self.model = to_cuda(self.model)

        # Define our optimizer. SGD = Stochastich Gradient Descent
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)

        # Load our dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = load_cifar10(self.batch_size)

        self.validation_check = len(self.dataloader_train) // 2

        # Tracking variables
        self.epoch_list = []
        self.training_step = 0
        self.TRAINING_STEP = []
        self.VALIDATION_LOSS = []
        self.TEST_LOSS = []
        self.TRAIN_LOSS = []
        self.TRAIN_ACC = []
        self.VALIDATION_ACC = []
        self.TEST_ACC = []

    def validation_epoch(self):
        """
            Computes the loss/accuracy for all three datasets.
            Train, validation and test.
        """
        self.model.eval()

        # Compute for training set
        train_loss, train_acc = compute_loss_and_accuracy(self.dataloader_train, self.model, self.loss_criterion)
        self.TRAIN_ACC.append(train_acc)
        self.TRAIN_LOSS.append(train_loss)

        # Compute for validation set
        validation_loss, validation_acc = compute_loss_and_accuracy(self.dataloader_val, self.model, self.loss_criterion)
        self.VALIDATION_ACC.append(validation_acc)
        self.VALIDATION_LOSS.append(validation_loss)

        # Compute for testing set
        test_loss, test_acc = compute_loss_and_accuracy( self.dataloader_test, self.model, self.loss_criterion)
        self.TEST_ACC.append(test_acc)
        self.TEST_LOSS.append(test_loss)
        print("Current validation loss:", validation_loss, " Accuracy:", validation_acc, "Test Accuracy:", test_acc)

        self.model.train()

    def should_early_stop(self):
        """
        Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        if len(self.VALIDATION_LOSS) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = self.VALIDATION_LOSS[-self.early_stop_count:]
        previous_loss = relevant_loss[0]
        for current_loss in relevant_loss[1:]:
            # If the next loss decrease, early stopping criteria is not met.
            if current_loss < previous_loss:
                return False
            previous_loss = current_loss
        return True

    def annealing_learning_rate(self):
        ratio = self.t/self.T
        self.learning_rate = self.a0/1+ratio


    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """
        # Track initial loss/accuracy
        if self.should_anneal:
            self.learning_rate = self.a0

        counter = 0
        self.TRAINING_STEP.append(self.training_step)
        self.validation_epoch()
        self.epoch_list.append(0)
        for epoch in range(self.epochs):
            print("Starting epoch", epoch+1)
            # Perform a full pass through all the training samples
            for batch_it, (X_batch, Y_batch) in enumerate(self.dataloader_train):
                self.training_step += 1
                # X_batch is the CIFAR10 images. Shape: [batch_size, 3, 32, 32]
                # Y_batch is the CIFAR10 image label. Shape: [batch_size]
                # Transfer images / labels to GPU VRAM, if possible
                X_batch = to_cuda(X_batch)
                Y_batch = to_cuda(Y_batch)

                # Perform the forward pass
                predictions = self.model(X_batch)
                # Compute the cross entropy loss for the batch
                loss = self.loss_criterion(predictions, Y_batch)

                # Backpropagation
                loss.backward()

                # Gradient descent step
                self.optimizer.step()

                # Reset all computed gradients to 0
                self.optimizer.zero_grad()
                 # Compute loss/accuracy for all three datasets.



                if batch_it % self.validation_check == 0:
                    counter += 1
                    self.TRAINING_STEP.append(self.training_step)
                    self.validation_epoch()
                    self.epoch_list.append( counter*(1/3) )         #three datapoints per epoch
                    # Check early stopping criteria.
                    if self.should_early_stop():
                        print("Early stopping at epoch", epoch)
                        print("Final validation_loss", self.VALIDATION_LOSS[-self.early_stop_count])
                        print("Final training loss", self.TRAIN_LOSS[-self.early_stop_count])
                        print("Final validation accuracy", self.VALIDATION_ACC[-self.early_stop_count])
                        print("Final test accuracy", self.TEST_ACC[-self.early_stop_count])
                        print("Final training accuracy", self.TRAIN_ACC[-self.early_stop_count])
                        return

                    if self.should_anneal:
                        self.t += 1
                        self.annealing_learning_rate()

        print("Early stopping at epoch", epoch)
        print("Final validation_loss", self.VALIDATION_LOSS[-1])
        print("Final training loss", self.TRAIN_LOSS[-1])
        print("Final validation accuracy", self.VALIDATION_ACC[-1])
        print("Final test accuracy", self.TEST_ACC[-1])
        print("Final training accuracy", self.TRAIN_ACC[-1])


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()

    os.makedirs("plots", exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(12, 8))
    plt.title("Cross Entropy Loss")
    plt.plot(trainer.epoch_list, trainer.VALIDATION_LOSS, label="Validation loss")
    plt.plot(trainer.epoch_list, trainer.TRAIN_LOSS, label="Training loss")
    plt.plot(trainer.epoch_list, trainer.TEST_LOSS, label="Testing Loss")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_loss_task3.png"))
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.title("Accuracy")
    plt.plot(trainer.epoch_list, trainer.VALIDATION_ACC, label="Validation Accuracy")
    plt.plot(trainer.epoch_list, trainer.TRAIN_ACC, label="Training Accuracy")
    plt.plot(trainer.epoch_list, trainer.TEST_ACC, label="Testing Accuracy")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_accuracy_task3.png"))
    plt.show()

    #print("Final test accuracy:", trainer.TEST_ACC[-trainer.early_stop_count])
    #print("Final validation accuracy:", trainer.VALIDATION_ACC[-trainer.early_stop_count])
