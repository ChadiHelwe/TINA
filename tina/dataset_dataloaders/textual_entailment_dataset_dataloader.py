from torch.utils.data import Dataset

from tina.preprocessing.utils import (
    read_dataset,
    read_dataset_te,
    read_dataset_te_negation_augmented,
    read_dataset_te_negation_augmented_unlikelihood_loss,
)


class TextualEntailmentDataset(Dataset):
    def __init__(self, path_dataset, size=None):
        super().__init__()
        self.data = read_dataset_te(path_dataset, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class TextualEntailmentNegationAugmentedDataset(Dataset):
    def __init__(self, path_dataset, size=None):
        super().__init__()
        self.data = read_dataset_te_negation_augmented(path_dataset, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class TextualEntailmentNegationAugmentedUnlikelihoodLossDataset(Dataset):
    def __init__(self, path_dataset, size=None):
        super().__init__()
        self.data = read_dataset_te_negation_augmented_unlikelihood_loss(
            path_dataset, size
        )

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class TextualEntailmentCombinedDataset(Dataset):
    def __init__(self, path_dataset, size=None):
        super().__init__()
        self.x, self.y = read_dataset(path_dataset, size)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
