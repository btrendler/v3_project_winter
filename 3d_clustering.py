import numpy as np
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, \
    adjusted_mutual_info_score, fowlkes_mallows_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt

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


# Convert a sample to X and y pairs, with numberic values in the place of NaNs
def to_xy_numeric(sample, labels):
    dep_col_idx = np.where(labels == DEP_COL)[0]
    X = np.delete(sample, dep_col_idx, axis=1)
    y = sample[:, dep_col_idx]
    y[np.isnan(y)] = 6
    return X, y


# Save a projection done with the given class to the file
def save_projection(X, y, filename, clazz, **kwargs):
    print(f"Doing training for {filename}...")
    proj = clazz(**kwargs)
    X_new = proj.fit_transform(X, y)
    np.save(f"projections/{filename}", X_new)
    print(f"Saved {filename}.")
    return X_new


if __name__ == "__main__":
    # Load in the partial dataset
    X, y = to_xy_numeric(*generate_sample(.1, random_state=19438))
    print(y.shape)
    # Do the TSNE projections & save
    for p in [2, 5, 10, 30, 50, 100, 200]:
        save_projection(X, y, f"tsne_perp{p}.npy", TSNE, n_components=2, perplexity=p, n_jobs=-1)
    # Do the UMAP unsupervised projections & save
    save_projection(X, None, "umap_unsup_2d.npy", UMAP, n_epochs=500, n_components=2, n_jobs=-1)
    save_projection(X, None, "umap_unsup_3d.npy", UMAP, n_epochs=500, n_components=3, n_jobs=-1)
    # Do supervised UMAP projections & save
    save_projection(X, y, "umap_sup_2d_def.npy", UMAP, n_epochs=500, n_components=2, n_jobs=-1)
    save_projection(X, y, "umap_sup_3d_def.npy", UMAP, n_epochs=500, n_components=3, n_jobs=-1)
    save_projection(X, y, "umap_sup_2d_grid.npy", UMAP, n_epochs=500, n_components=2, min_dist=1., n_neighbors=9, learning_rate=140.,
                    n_jobs=-1)
    save_projection(X, y, "umap_sup_3d_grid.npy", UMAP, n_epochs=500, n_components=3, min_dist=1., n_neighbors=9, learning_rate=140.,
                    n_jobs=-1)
    save_projection(X, y, "umap_sup_4d_grid.npy", UMAP, n_epochs=500, n_components=4, min_dist=1., n_neighbors=9, learning_rate=140.,
                    n_jobs=-1)

    # Load in the full dataset
    X, y = to_xy_numeric(*generate_sample(1., random_state=193))
    # Do the supervised UMAP projections for the full set
    save_projection(X, y, "umap_sup_2d_def_full.npy", UMAP, n_epochs=500, n_components=2, n_jobs=-1)
    save_projection(X, y, "umap_sup_3d_def_full.npy", UMAP, n_epochs=500, n_components=3, n_jobs=-1)
    save_projection(X, y, "umap_sup_2d_grid_full.npy", UMAP, n_epochs=500, n_components=2, min_dist=1., n_neighbors=9, learning_rate=140.,
                    n_jobs=-1)
    save_projection(X, y, "umap_sup_3d_grid_full.npy", UMAP, n_epochs=500, n_components=3, min_dist=1., n_neighbors=9, learning_rate=140.,
                    n_jobs=-1)

    # Load in a portion of the test data
    X, y = to_xy_numeric(*generate_sample(.1, set_name="test_patients", random_state=18478))
    # Do the supervised UMAP projections for the test set
    save_projection(X, y, "umap_sup_2d_def_test.npy", UMAP, n_epochs=500, n_components=2, n_jobs=-1)
    save_projection(X, y, "umap_sup_3d_def_test.npy", UMAP, n_epochs=500, n_components=3, n_jobs=-1)
    save_projection(X, y, "umap_sup_2d_grid_test.npy", UMAP, n_epochs=500, n_components=2, min_dist=1., n_neighbors=9, learning_rate=140.,
                    n_jobs=-1)
    save_projection(X, y, "umap_sup_3d_grid_test.npy", UMAP, n_epochs=500, n_components=3, min_dist=1., n_neighbors=9, learning_rate=140.,
                    n_jobs=-1)
