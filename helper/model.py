import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm


class MyModel:

    def __init__(
        self,
        nb_classes: int,
        device: torch.device,
        leaning_rate: float = 0.01,
        momentum: float = 0.9,
        mode: str = "max",
        patience: int = 3,
        threshold: float = 0.9,
    ):
        self.nb_classes = nb_classes
        self.device = device

        # load resnet pretrain dataset
        self.model = models.resnet18(pretrained=True)

        # changement du dernier layer
        self.model.fc = nn.Linear(self.model.fc.in_features, self.nb_classes)
        self.model = self.model.to(device)

        # loss function, optimizer and scheduler defintion
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=leaning_rate, momentum=momentum
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode=mode, patience=patience, threshold=threshold
        )

    def eval(self, test_loader: DataLoader, num_augmentations: int = 1) -> float:
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for data in tqdm(test_loader):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0) * num_augmentations
                correct += (predicted == labels.repeat(num_augmentations)).sum().item()
        test_acc = 100.0 * correct / total
        print("Accuracy on test dataset : %.2f" % (test_acc))
        return test_acc

    def train(
        self, train_loader: DataLoader, test_loader: DataLoader, nb_epochs: int = 10
    ) -> Tuple[List[float]]:
        losses: List[float] = []
        accuracies: List[float] = []
        test_accuracies: List[float] = []

        self.model.train()

        for epoch in range(nb_epochs):
            since = time.time()
            running_loss = 0.0
            running_correct = 0.0

            for data in tqdm(train_loader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                _, preds = torch.max(outputs.data, 1)

                loss = self.criterion(outputs, labels)
                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()
                running_correct += (labels == preds).sum().item()

            epoch_duration = time.time() - since
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 / 32 * running_correct / len(train_loader)
            print(
                f"Epoch {epoch + 1}, duration: {epoch_duration}s, loss: %.4f, acc: %.4f"
                % (epoch_loss, epoch_acc)
            )

            losses.append(epoch_loss)
            accuracies.append(epoch_acc)

            self.model.eval()
            test_acc = self.eval(test_loader)
            test_accuracies.append(test_acc)

            self.model.train()
            self.scheduler.step(test_acc)
            since = time.time()

        print("End training")
        return losses, accuracies, test_accuracies

    def compute_true_predicted_labels(
        self, test_loader: DataLoader
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        true_labels: List[np.ndarray] = []
        predicted_labels: List[np.ndarray] = []

        with torch.no_grad():
            for images, labels in tqdm(
                test_loader, desc="Compute true labels and predicted labels"
            ):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                _, predicted = torch.max(outputs.data, 1)

                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())

        return true_labels, predicted_labels
