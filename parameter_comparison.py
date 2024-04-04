import numpy as np
from umap import UMAP
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 200

import warnings
warnings.filterwarnings("ignore")

DEP_COL = "_HYPNO-mode"

# Utility functions to load in the data
dataset = np.load("sc-agg-f16.npz")


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
    return X, y


# Configure the combinations of parameters we'll search
PARAM_COMBOS = [
    ("Epochs", "n_epochs", [500, 700, 900, 1100, 1300]),
    ("Metric", "metric", ["euclidean", "chebyshev", "braycurtis", "euclidean", "minkowski", "mahalanobis"]),
    ("Neighbors", "n_neighbors", [10,30,50,70,90]),
    ("Learning Rate", "learning_rate", [1.0, 10.0, 100.0, 0.1]),
    ("Initialization", "init", ["spectral", "random", "pca", "tswspectral"]),
    ("Op Mix Ratio", "set_op_mix_ratio", [1., .75, .3, 0.]),
    ("Local Connectivity", "local_connectivity", [1,5,10,15,20]),
    ("Repulsion Strength", "repulsion_strength", [1., 2., 5., 10., 0.3]),
    ("Transform Queue Size", "transform_queue_size", [4., 2., 1., 6.]),
    ("Target Weight", "target_weight", [0.5, 0.1, 0.8, 0.9, 1.]),
]

# Load in a few random portions of the data
data_train = to_xy_numeric(*generate_sample(1., random_state=1919))
data_valid = to_xy_numeric(*generate_sample(1., set_name="validate_patients", random_state=19))


def compute_one_set(title, parameter, p_values):
    print(f"Beginning computation for {title}...")

    # Create parameter dictionary
    params = {"verbose": True, "n_components": 2, "n_jobs": -1, "n_epochs": 900}

    # Loop through the parameter values
    results = []
    for p_value in p_values:
        params[parameter] = p_value
        model = UMAP(**params)
        train_res = model.fit_transform(*data_train)
        valid_res = model.transform(data_valid[0])
        results.append((train_res, valid_res))

    print("Finished.")
    return title, parameter, p_values, results


def plot_set(title, parameter, p_values, results):
    cols = len(p_values)
    plt.figure(figsize=(4 * cols, 4 * 2))

    for i, (p_value, (train_res, valid_res)) in enumerate(zip(p_values, results)):
        plt.subplot(2, cols, 1 + i)
        plt.scatter(*train_res.T, c=data_train[1].ravel(), cmap="rainbow", marker=".", alpha=0.1)
        plt.title(f"{parameter} = {p_value}")
        plt.subplot(2, cols, 1 + i + cols)
        plt.scatter(*valid_res.T, c=data_valid[1].ravel(), cmap="rainbow", marker=".", alpha=0.1)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"parameter-comparison/{title}.png")


if __name__ == "__main__":
    for combo in PARAM_COMBOS:
        try:
            plot_set(*compute_one_set(*combo))
        except Exception as e:
            print(f"Error on combo {combo}: {e}")