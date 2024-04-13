import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 200

import warnings
warnings.filterwarnings("ignore")

DEP_COL = "_HYPNO-mode"

# Utility functions to load in the data
dataset = np.load("sc-agg-f16-veryhighdim.npz")


# Create a reproducible sample of rows from the given set
def generate_sample(size_per, set_name="train_patients", random_state=42):
    # Set up the random state & the row storage
    r = np.random.RandomState(random_state)
    rows = []

    # Loop through each person in the named set and select that number of rows, randomly
    for entry in dataset[set_name]:
        entry_set = dataset[entry]
        to_get = size_per if type(size_per) is int else int(size_per * entry_set.shape[0])
        selection = r.choice(range(entry_set.shape[0]), size=to_get, replace=False)
        rows.extend(entry_set[selection])

    # Return the sample and the labels
    return np.array(rows, dtype=float), dataset["labels"]


# Convert a sample to X and y pairs, with numeric values in the place of NaNs
def to_xy_numeric(sample, labels):
    dep_col_idx = np.where(labels == DEP_COL)[0]
    X = np.delete(sample, dep_col_idx, axis=1)
    y = sample[:, dep_col_idx]
    y[np.isnan(y)] = 6
    return X, y.ravel()


# Load in a few random portions of the data
data_train = to_xy_numeric(*generate_sample(1., random_state=1919))
data_valid = to_xy_numeric(*generate_sample(1., set_name="validate_patients", random_state=19))
data_test_ = to_xy_numeric(*generate_sample(1., set_name="test_patients", random_state=19))


if __name__ == "__main__":
    rfc = RandomForestClassifier() #n_jobs=-1, criterion="gini", max_depth=7, max_features=0.5, min_samples_leaf=10, n_estimators=60)
    gs = GridSearchCV(rfc, {
        "n_estimators": [100, 120, 140], # 80, 120
        "max_depth": [7, 9, 11], #4, 6
        "min_samples_leaf": [5, 15], # 5
        "max_features": ["sqrt", .5], # .5
        "criterion": ["gini"]
    }, verbose=10, n_jobs=-1, cv=2)
    # gs.fit(*data_train)
    gs.fit(*data_train)
    print("Fitting complete:")
    print(gs.best_params_)
    print(gs.best_score_)
    print(gs.score(*data_valid))
    print(gs.score(*data_test_))