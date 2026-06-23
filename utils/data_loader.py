import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class AirQualityDataset(Dataset):
    def __init__(self, features_csv, targets_csv, seq_len=3):
        self.seq_len = seq_len
        self.X = pd.read_csv(features_csv).values.astype('float32')
        self.y = pd.read_csv(targets_csv).values.astype('float32')
        self.X = torch.tensor(self.X)
        self.y = torch.tensor(self.y)

    def __len__(self):
        # Number of sequences available
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        X_seq = self.X[idx : idx + self.seq_len]   # shape: (seq_len, input_size)
        y_seq = self.y[idx + self.seq_len - 1]    # predict only last step
        return X_seq, y_seq

def get_data_loaders(config, batch_size=32):
    # Use os.path.join to form file paths
    X_train_path = os.path.join(config['data']['processed'], 'X_train_norm.csv')
    y_train_path = os.path.join(config['data']['processed'], 'y_train_norm.csv')
    X_test_path = os.path.join(config['data']['processed'], 'X_test_norm.csv')
    y_test_path = os.path.join(config['data']['processed'], 'y_test_norm.csv')

    train_dataset = AirQualityDataset(X_train_path, y_train_path, seq_len=3)
    test_dataset = AirQualityDataset(X_test_path, y_test_path, seq_len=3)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = get_data_loaders(config)
    for batch_idx, (features, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}: features shape {features.shape}, targets shape {targets.shape}")
        break

