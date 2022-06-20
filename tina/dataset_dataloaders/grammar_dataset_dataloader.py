from torch.utils.data import Dataset

from tina.preprocessing.utils import read_dataset_grammar


class GrammarDataset(Dataset):
    def __init__(self, path_dataset):
        super().__init__()
        self.p, self.h, self.n_p, self.n_h, self.l = read_dataset_grammar(path_dataset)

    def __getitem__(self, index):
        return (
            self.p[index],
            self.h[index],
            self.n_p[index],
            self.n_h[index],
            self.l[index],
        )

    def __len__(self):
        return len(self.p)
