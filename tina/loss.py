import torch
from torch import nn


class CrossEntropyAndUnlikelihoodLoss(nn.Module):
    def __init__(self, ignore_index=-100, epsilon=1e-10) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def forward(self, pred_values, target_values, known_labels, unknown_labels):
        """
        The function takes in the predicted values, the target values, the known labels, and the unknown
        labels. It then calculates the loss for each prediction and target pair

        :param pred_values: the output of the model, a tensor of shape (batch_size, num_classes)
        :param target_values: the ground truth labels
        :param known_labels: a tensor of shape (batch_size, num_classes) where each element is either 0
        or 1
        :param unknown_labels: a tensor of shape (batch_size, num_classes)
        :return: The loss is being returned.
        """
        loss = 0.0
        n, _ = pred_values.shape
        for pred, target, known_label, unknown_label in zip(
            pred_values, target_values, known_labels, unknown_labels
        ):
            class_index = int(target.item())
            if class_index == self.ignore_index:
                n -= 1
                continue
            prob_pred = torch.exp(pred[class_index]) / (torch.exp(pred).sum())
            loss = (
                loss
                + torch.log(prob_pred) * known_label
                + torch.log(1 - prob_pred + self.epsilon) * unknown_label
            )
        loss = -loss / n
        return loss

    def __call__(self, pred_values, target_values, known_labels, unknown_labels):
        """
        The function takes in the predicted values, the target values, the known labels, and the unknown
        labels, and returns the loss

        :param pred_values: the output of the model, a tensor of shape (batch_size, num_classes)
        :param target_values: the ground truth labels
        :param known_labels: a tensor of shape (batch_size, num_classes) where each element is either 0
        or 1
        :param unknown_labels: a tensor of shape (batch_size, num_classes)
        :return: The loss is being returned.
        """
        return self.forward(pred_values, target_values, known_labels, unknown_labels)
