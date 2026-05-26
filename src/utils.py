import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score,
                             precision_recall_curve)


def train_model(model, train_loader, criterion, optimizer,
                num_epochs: int = 30, device: str = 'cpu') -> list:
    """
    Autoencoder'ı eğitir.

    Returns:
        train_losses: Her epoch'un ortalama MSE kaybı listesi
    """
    train_losses = []

    print(f"Eğitim başlıyor... ({num_epochs} epoch, device: {device})\n")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            batch  = batch.to(device)
            output = model(batch)
            loss   = criterion(output, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1:2d}/{num_epochs}] | Loss: {avg_loss:.6f}")

    print("\nEğitim tamamlandı!")
    return train_losses


def compute_reconstruction_errors(model, data_loader, device: str = 'cpu') -> np.ndarray:
    """
    Test seti üzerinde her örnek için reconstruction error hesaplar.

    Returns:
        numpy array of per-sample MSE errors
    """
    model.eval()
    errors = []

    with torch.no_grad():
        for batch in data_loader:
            batch  = batch.to(device)
            output = model(batch)
            mse    = torch.mean((output - batch) ** 2, dim=1)
            errors.extend(mse.cpu().numpy())

    return np.array(errors)


def find_best_threshold(errors: np.ndarray, y_true: np.ndarray,
                        percentile_range=(80, 100), step=0.5) -> tuple:
    """
    F1 skorunu maksimize eden threshold'u bulur.

    Returns:
        best_threshold (float), best_f1 (float)
    """
    thresholds = np.percentile(errors, np.arange(*percentile_range, step))
    f1_scores  = [f1_score(y_true, (errors > t).astype(int))
                  for t in thresholds]

    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def evaluate(errors: np.ndarray, y_true: np.ndarray, threshold: float):
    """
    Verilen threshold ile tam değerlendirme raporu yazdırır.
    """
    y_pred = (errors > threshold).astype(int)
    auc    = roc_auc_score(y_true, errors)

    print(f"Threshold  : {threshold:.4f}")
    print(f"ROC-AUC    : {auc:.4f}")
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred,
                                target_names=['Normal', 'Fraud']))
    return y_pred, auc


def plot_training_loss(train_losses: list, save_path: str = None):
    """Training loss eğrisini çizer."""
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses,
             color='steelblue', linewidth=2)
    plt.title('Training Loss (Reconstruction Error)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_evaluation(errors, y_true, y_pred, threshold, auc,
                    save_path: str = None):
    """4'lü değerlendirme grafiği: hata dağılımı, CM, ROC, PR."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Reconstruction Error Dağılımı
    axes[0, 0].hist(errors[y_true == 0], bins=100, alpha=0.6,
                    color='steelblue', label='Normal', density=True)
    axes[0, 0].hist(errors[y_true == 1], bins=100, alpha=0.6,
                    color='crimson', label='Fraud', density=True)
    axes[0, 0].axvline(threshold, color='orange', linestyle='--',
                       linewidth=2, label=f'Threshold={threshold:.3f}')
    axes[0, 0].set_title('Reconstruction Error Distribution')
    axes[0, 0].set_xlabel('MSE')
    axes[0, 0].legend()

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    axes[0, 1].set_title('Confusion Matrix')
    axes[0, 1].set_ylabel('Gerçek')
    axes[0, 1].set_xlabel('Tahmin')

    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, errors)
    axes[1, 0].plot(fpr, tpr, color='steelblue', linewidth=2,
                    label=f'AUC = {auc:.4f}')
    axes[1, 0].plot([0, 1], [0, 1], '--', color='gray')
    axes[1, 0].set_title('ROC Curve')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].legend()

    # 4. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, errors)
    axes[1, 1].plot(recall, precision, color='crimson', linewidth=2)
    axes[1, 1].set_title('Precision-Recall Curve')
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_ylabel('Precision')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
