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
    return proj


def save_prediction(model, X, y, filename):
    print(f"Doing predicting for {filename}...")
    X_new = model.transform(X)
    np.save(f"projections/{filename}", X_new)
    print(f"Saved {filename}.")


def make_projections():
    # Load in the partial dataset
    X, y = to_xy_numeric(*generate_sample(.1, random_state=19438))
    np.save("projections/y_part.npy", y)
    # Do the TSNE projections & save
    # for p in [2, 5, 10, 30, 50, 100, 200]:
    #     save_projection(X, y, f"tsne_perp{p}.npy", TSNE, n_components=2, perplexity=p, n_jobs=-1)
    # Do the UMAP unsupervised projections & save
    save_projection(X, None, "umap_unsup_2d.npy", UMAP, n_epochs=500, n_components=2, n_jobs=-1, n_neighbors=50)
    save_projection(X, None, "umap_unsup_3d.npy", UMAP, n_epochs=500, n_components=3, n_jobs=-1, n_neighbors=50)
    # Do supervised UMAP projections & save
    save_projection(X, y, "umap_sup_2d_def.npy", UMAP, n_epochs=500, n_components=2, n_jobs=-1, n_neighbors=50)
    save_projection(X, y, "umap_sup_3d_def.npy", UMAP, n_epochs=500, n_components=3, n_jobs=-1, n_neighbors=50)
    save_projection(X, y, "umap_sup_2d_grid.npy", UMAP, n_epochs=500, n_components=2, min_dist=1., n_neighbors=9,
                    learning_rate=140.,
                    n_jobs=-1)
    save_projection(X, y, "umap_sup_3d_grid.npy", UMAP, n_epochs=500, n_components=3, min_dist=1., n_neighbors=9,
                    learning_rate=140.,
                    n_jobs=-1)
    save_projection(X, y, "umap_sup_4d_grid.npy", UMAP, n_epochs=500, n_components=4, min_dist=1., n_neighbors=9,
                    learning_rate=140.,
                    n_jobs=-1)

    # Load in the full dataset
    X, y = to_xy_numeric(*generate_sample(1., random_state=193))
    np.save("projections/y_full.npy", y)
    # Do the supervised UMAP projections for the full set
    sup_2d_def = save_projection(X, y, "umap_sup_2d_def_full.npy", UMAP, n_epochs=500, n_components=2, n_jobs=-1, n_neighbors=50)
    sup_3d_def = save_projection(X, y, "umap_sup_3d_def_full.npy", UMAP, n_epochs=500, n_components=3, n_jobs=-1, n_neighbors=50)
    sup_2d_grid = save_projection(X, y, "umap_sup_2d_grid_full.npy", UMAP, n_epochs=500, n_components=2, min_dist=1., n_neighbors=9,
                    learning_rate=140.,
                    n_jobs=-1)
    sup_3d_grid = save_projection(X, y, "umap_sup_3d_grid_full.npy", UMAP, n_epochs=500, n_components=3, min_dist=1., n_neighbors=9,
                    learning_rate=140.,
                    n_jobs=-1)

    # Load in a portion of the test data
    X, y = to_xy_numeric(*generate_sample(.1, set_name="test_patients", random_state=18478))
    np.save("projections/y_test.npy", y)
    # Do the supervised UMAP projections for the test set
    save_prediction(sup_2d_def, X, y, "umap_sup_2d_def_test.npy")
    save_prediction(sup_3d_def, X, y, "umap_sup_3d_def_test.npy")
    save_prediction(sup_2d_grid, X, y, "umap_sup_2d_grid_test.npy")
    save_prediction(sup_3d_grid, X, y, "umap_sup_3d_grid_test.npy")


def make_2d_group_plot(files_in, y_file_in, file_out, suptitle, titles, dimensions, shape):
    # Read in the data
    X = [np.load(f"projections/{file_in}.npy") for file_in in files_in]
    y = np.load(f"projections/{y_file_in}.npy")

    # Make each of the subplots
    plt.figure(figsize=dimensions)
    for i, (x, title) in enumerate(zip(X, titles)):
        plt.subplot(*shape, i + 1)
        plt.title(title)
        plt.scatter(*x.T, c=y, cmap="rainbow", marker=".")
    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(f"figures/proj-{file_out}.pdf")
    plt.show()


def make_2d_plot(file_in, y_file_in, title, dimensions):
    # Read in the data
    X = np.load(f"projections/{file_in}.npy")
    y = np.load(f"projections/{y_file_in}.npy")

    # Make the scatter plot and save to the file
    plt.figure(figsize=dimensions)
    plt.title(title)
    plt.scatter(*X.T, c=y, cmap="rainbow", marker=".")
    plt.tight_layout()
    plt.savefig(f"figures/proj-{file_in}.pdf")
    plt.show()


def make_3d_plot(file_in, y_file_in, title, dimensions):
    # Read in the data
    X = np.load(f"projections/{file_in}.npy")
    y = np.load(f"projections/{y_file_in}.npy")

    # Make the scatter plot and save to the file
    fig = plt.figure(figsize=dimensions)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*X.T, c=y, cmap="rainbow", marker=".")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"figures/proj-{file_in}.pdf")
    plt.show()


def make_4d_plot(file_in, y_file_in, title, dimensions):
    # Read in the data
    X = np.load(f"projections/{file_in}.npy")
    y = np.load(f"projections/{y_file_in}.npy")

    # Make the scatter plot and save to the file
    plt.figure(figsize=dimensions)
    plt.subplot(131)
    plt.title("Comp 1 v. Comp 2")
    plt.scatter(*X.T[[0, 1]], c=y, cmap="rainbow", marker=".")
    plt.subplot(132)
    plt.title("Comp 1 v. Comp 3")
    plt.scatter(*X.T[[0, 2]], c=y, cmap="rainbow", marker=".")
    plt.subplot(133)
    plt.title("Comp 1 v. Comp 4")
    plt.scatter(*X.T[[0, 3]], c=y, cmap="rainbow", marker=".")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"figures/proj-{file_in}.pdf")
    plt.show()


if __name__ == "__main__":
    SIZE_FACTOR = 2
    #make_projections()

    # Make the TSNE plot
    perps = [2, 5, 10, 30, 50, 100]
    make_2d_group_plot([f"tsne_perp{p}" for p in perps], "y_part", "tsne",
                       ''"TSNE Trained on 10% of the Test Data", [f"Perplexity: {p}" for p in perps], (SIZE_FACTOR * 3, SIZE_FACTOR * 2), (2, 3))

    # Make the 2D UMAP figures
    make_2d_plot("umap_unsup_2d", "y_part", "Unsupervised UMAP in 2D\n(10% of training data, 50 neighbors, default params)", (SIZE_FACTOR * 2, SIZE_FACTOR * 2))
    make_2d_plot("umap_sup_2d_def", "y_part", "Supervised UMAP in 2D\n(10% of training data, 50 neighbors, default params)", (SIZE_FACTOR * 2, SIZE_FACTOR * 2))
    make_2d_plot("umap_sup_2d_grid", "y_part", "Supervised UMAP in 2D\n(10% of training data, grid-searched params)", (SIZE_FACTOR * 2, SIZE_FACTOR * 2))
    make_2d_plot("umap_sup_2d_def_full", "y_full", "Supervised UMAP in 2D\n(Training data, 50 neighbors, default params)", (SIZE_FACTOR * 2, SIZE_FACTOR * 2))
    make_2d_plot("umap_sup_2d_grid_full", "y_full", "Supervised UMAP in 2D\n(Training data, grid-searched params)", (SIZE_FACTOR * 2, SIZE_FACTOR * 2))
    make_2d_plot("umap_sup_2d_def_test", "y_test", "Supervised UMAP in 2D\n(Test data, 50 neighbors, default params)", (SIZE_FACTOR * 2, SIZE_FACTOR * 2))
    make_2d_plot("umap_sup_2d_grid_test", "y_test", "Supervised UMAP in 2D\n(Test data, grid-searched params)", (SIZE_FACTOR * 2, SIZE_FACTOR * 2))

    # Make the 3D UMAP figures
    make_3d_plot("umap_unsup_3d", "y_part", "Unsupervised UMAP in 3D\n(50 neighbors, default params)\n(10% of training data)", (SIZE_FACTOR * 2, SIZE_FACTOR * 2))
    make_3d_plot("umap_sup_3d_def", "y_part", "Supervised UMAP in 3D\n(10% of training data, 50 neighbors, default params)", (SIZE_FACTOR * 2, SIZE_FACTOR * 2))
    make_3d_plot("umap_sup_3d_grid", "y_part", "Supervised UMAP in 3D\n(10% of training data, grid-searched params)", (SIZE_FACTOR * 2, SIZE_FACTOR * 2))
    make_3d_plot("umap_sup_3d_def_full", "y_full", "Supervised UMAP in 3D\n(Training data, 50 neighbors, default params)", (SIZE_FACTOR * 2, SIZE_FACTOR * 2))
    make_3d_plot("umap_sup_3d_grid_full", "y_full", "Supervised UMAP in 3D\n(Training data, grid-searched params)", (SIZE_FACTOR * 2, SIZE_FACTOR * 2))
    make_3d_plot("umap_sup_3d_def_test", "y_test", "Supervised UMAP in 3D\n(Test data, 50 neighbors, default params)", (SIZE_FACTOR * 2, SIZE_FACTOR * 2))
    make_3d_plot("umap_sup_3d_grid_test", "y_test", "Supervised UMAP in 3D\n(Test data, grid-searched params)", (SIZE_FACTOR * 2, SIZE_FACTOR * 2))

    # Make the 4D UMAP figure
    make_4d_plot("umap_sup_4d_grid", "y_part", "Supervised UMAP into 4D\n(using parameters found via GridSearch)",
                (15, 5))
