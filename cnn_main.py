import os
import numpy as np
import mne
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import TensorDataset, DataLoader

DATA_FOLDER = "data"
RUN_FILES = [
    "S001R04.edf",
    "S001R08.edf",
    "S001R12.edf"
]

TARGET_EVENTS = ["T1", "T2"]

TMIN = 0.5
TMAX = 4.0
LOW_FREQ = 8.0
HIGH_FREQ = 30.0
RANDOM_STATE = 42
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001


def load_run(filepath):
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    raw.filter(LOW_FREQ, HIGH_FREQ, fir_design="firwin", verbose=False)

    events, event_dict = mne.events_from_annotations(raw, verbose=False)

    available_event_id = {}
    for key in TARGET_EVENTS:
        if key in event_dict:
            available_event_id[key] = event_dict[key]

    if len(available_event_id) < 2:
        print(f"Skipping {os.path.basename(filepath)}: T1/T2 not both found.")
        return None, None

    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")

    epochs = mne.Epochs(
        raw,
        events,
        event_id=available_event_id,
        tmin=TMIN,
        tmax=TMAX,
        picks=picks,
        baseline=None,
        preload=True,
        verbose=False
    )

    X = epochs.get_data()
    y = epochs.events[:, -1]

    label_map = {
        available_event_id["T1"]: 0,
        available_event_id["T2"]: 1
    }
    y = np.array([label_map[val] for val in y])

    return X, y


def load_all_runs():
    X_list = []
    y_list = []

    for fname in RUN_FILES:
        path = os.path.join(DATA_FOLDER, fname)

        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue

        X, y = load_run(path)
        if X is None:
            continue

        X_list.append(X)
        y_list.append(y)
        print(f"Loaded {fname}: X={X.shape}, y={y.shape}")

    if not X_list:
        raise ValueError("No usable data loaded.")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    return X_all, y_all


class SimpleEEGCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 25), padding=(0, 12)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),

            nn.Conv2d(8, 16, kernel_size=(64, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 1 * 35, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    X, y = load_all_runs()

    # normalize each trial
    X = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-8)

    # reshape for CNN: (samples, 1, channels, time)
    X = X[:, np.newaxis, :, :]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = SimpleEEGCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    model.eval()
    preds = []
    true = []

    with torch.no_grad():
        for xb, yb in test_loader:
            outputs = model(xb)
            predicted = torch.argmax(outputs, dim=1)
            preds.extend(predicted.numpy())
            true.extend(yb.numpy())

    acc = accuracy_score(true, preds)
    print("\nCNN Accuracy:", round(acc, 4))
    print("\nClassification Report:")
    print(classification_report(true, preds, target_names=["left", "right"]))


if __name__ == "__main__":
    main()