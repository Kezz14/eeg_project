import os
import numpy as np
import mne

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from mne.decoding import CSP

# -----------------------------
# SETTINGS
# -----------------------------
DATA_FOLDER = "data"

# Start with one subject's imagery runs:
# Replace these filenames with your real EDF filenames
RUN_FILES = [
    "S001R04.edf",
    "S001R08.edf",
    "S001R12.edf"
]

# Event mapping for runs 4, 8, 12:
# T1 = left fist imagery
# T2 = right fist imagery
EVENT_ID = {
    "T1": 1,
    "T2": 2
}

TMIN = 0.5   # seconds after cue onset
TMAX = 4.0   # seconds after cue onset
LOW_FREQ = 8.0
HIGH_FREQ = 30.0
RANDOM_STATE = 42


# -----------------------------
# LOAD ONE EDF FILE
# -----------------------------
def load_run(filepath):
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)

    # Bandpass filter for motor imagery rhythms
    raw.filter(LOW_FREQ, HIGH_FREQ, fir_design="firwin", verbose=False)

    # Extract events from annotations
    events, event_dict = mne.events_from_annotations(raw, verbose=False)

    # Keep only T1 and T2 if present
    available_event_id = {}
    for key in EVENT_ID:
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

    X = epochs.get_data()  # shape: (n_trials, n_channels, n_times)
    y = epochs.events[:, -1]

    # Convert event labels to 0 and 1
    # event code for T1 -> 0
    # event code for T2 -> 1
    label_map = {
        available_event_id["T1"]: 0,
        available_event_id["T2"]: 1
    }
    y = np.array([label_map[val] for val in y])

    return X, y


# -----------------------------
# LOAD ALL RUNS
# -----------------------------
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
        raise ValueError("No usable data loaded. Check filenames and data folder.")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    return X_all, y_all


# -----------------------------
# MAIN
# -----------------------------
def main():
    X, y = load_all_runs()

    print("\nFinal dataset shape:")
    print("X:", X.shape)
    print("y:", y.shape)
    print("Class counts:", np.bincount(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    clf = Pipeline([
        ("csp", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
        ("svm", SVC(kernel="linear"))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", round(acc, 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["left", "right"]))


if __name__ == "__main__":
    main()