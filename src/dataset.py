import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class FraudDataset(Dataset):
    """
    Kredi kartı işlemleri için PyTorch Dataset sınıfı.

    Args:
        dataframe (pd.DataFrame): Feature'ları içeren DataFrame.
                                  Label sütunu olmamalı.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.data = torch.tensor(dataframe.values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def load_and_split(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    CSV'yi yükler, ön işler ve train/test setlerine böler.

    Strateji:
        - Train: Yalnızca normal işlemler (model bunları öğrenir)
        - Test : Normal + Fraud karışık (model burada değerlendirilir)

    Returns:
        X_train      : Normal işlemlerin %80'i (DataFrame)
        X_test       : Normal %20 + tüm fraud (DataFrame, 'label' sütunlu)
        y_test_labels: Test setinin gerçek etiketleri (numpy array)
    """
    df = pd.read_csv(csv_path)

    # Time sütununu düşür
    df = df.drop(columns=['Time'])

    # Amount'u normalize et
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])

    # Normal ve Fraud'u ayır
    normal = df[df['Class'] == 0].drop(columns=['Class'])
    fraud  = df[df['Class'] == 1].drop(columns=['Class'])

    # Train: normal işlemlerin %80'i
    X_train, X_val = train_test_split(normal, test_size=test_size,
                                      random_state=random_state)

    # Test: normal %20 + fraud
    X_val_labeled         = X_val.copy()
    X_val_labeled['label'] = 0
    fraud_labeled          = fraud.copy()
    fraud_labeled['label'] = 1

    X_test = pd.concat([X_val_labeled, fraud_labeled], ignore_index=True)
    X_test = X_test.sample(frac=1, random_state=random_state).reset_index(drop=True)

    y_test_labels = X_test['label'].values
    X_test        = X_test.drop(columns=['label'])

    return X_train, X_test, y_test_labels


def get_dataloaders(X_train: pd.DataFrame, X_test: pd.DataFrame,
                    batch_size: int = 256):
    """
    Train ve Test DataLoader'larını oluşturur.

    Returns:
        train_loader, test_loader
    """
    train_loader = DataLoader(FraudDataset(X_train),
                              batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(FraudDataset(X_test),
                              batch_size=batch_size, shuffle=False)
    return train_loader, test_loader