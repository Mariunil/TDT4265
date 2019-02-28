import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from dataloaders import load_cifar10
from utils import to_cuda, compute_loss_and_accuracy

from task3_plotter import file_reader

# github: https://github.com/Mariunil/TDT4265.git

class ExampleModel(nn.Module):                  #Model 2

    def __init__(self, image_channels, num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        num_filters = 32  # Set number of filters in first conv layer
        num_units_dense_Relu = 64

        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d( in_channels=image_channels,  out_channels=num_filters, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d( in_channels=num_filters,  out_channels=num_filters*2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d( in_channels=num_filters*2,  out_channels=num_filters*4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 8*16*16
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, num_units_dense_Relu),
            nn.BatchNorm1d(num_units_dense_Relu),
            nn.ReLU(),
            nn.Linear(num_units_dense_Relu, num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """

        # Run image through convolutional layers
        x = self.feature_extractor(x)
        #print(x.shape)
        # Reshape our input to (batch_size, num_output_features)
        x = x.view(-1, self.num_output_features)
        # Forward pass through the fully-connected layers.
        x = self.classifier(x)
        return x

class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

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
        self.L2 = 0.001
        self.nesterov = False
        self.early_stop_count = 1000
        self.should_anneal = False
        self.T = 5
        self.t = 0
        self.a0 = 5e-2

        # Architecture

        # Since we are doing multi-class classification, we use the CrossEntropyLoss
        self.loss_criterion = nn.CrossEntropyLoss()
        # Initialize the mode
        self.model = ExampleModel(image_channels=3, num_classes=10)
        # Transfer model to GPU VRAM, if possible.
        self.model = to_cuda(self.model)

        # Define our optimizer. SGD = Stochastich Gradient Descent
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay = self.L2)

        # Load our dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = load_cifar10(self.batch_size)

        self.validation_check = len(self.dataloader_train) // 2

        # Tracking variables
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

        self.TRAINING_STEP.append(self.training_step)
        self.validation_epoch()
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
                    self.TRAINING_STEP.append(self.training_step)
                    self.validation_epoch()
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

    task3_test_acc, task3_val_acc, task3_train_acc, task3_test_loss, task3_val_loss, task3_train_loss = file_reader()
    epoch_list = [0, 0.33, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3, 3.33, 3.66, 4, 4.33, 4.66, 5]
    os.makedirs("plots", exist_ok=True)

    #----------------------Loss plot--------------------------------------#
    plt.figure(figsize=(12, 8))
    plt.title("Cross Entropy Loss")

    plt.plot(epoch_list, trainer.VALIDATION_LOSS, label="Our model - Validation loss")
    plt.plot(epoch_list, trainer.TRAIN_LOSS, label="Our model - Training loss")
    plt.plot(epoch_list, trainer.TEST_LOSS, label="Our model - Testing Loss")

    plt.plot(epoch_list, task3_val_loss, label="Resnet18 - Validation loss")
    plt.plot(epoch_list, task3_train_loss, label="Resnet18 - Training loss")
    plt.plot(epoch_list, task3_test_loss, label="Resnet18 - Testing Loss")

    plt.legend()
    plt.savefig(os.path.join("plots", "final_loss_doubleplot.png"))
    plt.show()
    #----------------------------------------------------------------------#

    #----------------------Accuracy plot-----------------------------------#
    plt.figure(figsize=(12, 8))
    plt.title("Accuracy")
    plt.plot(epoch_list, trainer.VALIDATION_ACC, label="Our model - Validation Accuracy")
    plt.plot(epoch_list, trainer.TRAIN_ACC, label="Our model - Training Accuracy")
    plt.plot(epoch_list, trainer.TEST_ACC, label="Our model - Testing Accuracy")

    plt.plot(epoch_list, task3_val_acc, label="Resnet18 - Validation Accuracy")
    plt.plot(epoch_list, task3_train_acc, label="Resnet18 - Training Accuracy")
    plt.plot(epoch_list, task3_test_acc, label="Resnet18 - Testing Accuracy")

    plt.legend()
    plt.savefig(os.path.join("plots", "final_accuracy_doubleplot.png"))
    plt.show()
    #----------------------------------------------------------------------#

    #print("Final test accuracy:", trainer.TEST_ACC[-trainer.early_stop_count])
    #print("Final validation accuracy:", trainer.VALIDATION_ACC[-trainer.early_stop_count])
