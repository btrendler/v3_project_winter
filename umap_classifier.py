import time
from sklearn.model_selection import ParameterGrid
from pynndescent import NNDescent
from sklearn.ensemble import RandomForestClassifier
from umap import UMAP
from sklearn.metrics import v_measure_score, accuracy_score
import numpy as np
from tqdm import tqdm


class UMAPClassifier:
    def __init__(self, X, y, n_neighbors, input_length=0, max_candidates=60, n_jobs=-1, verbose=False, umap_kwargs=dict(), rfc_kwargs=dict()):
        # Precompute the distances
        self.index = NNDescent(
            np.vstack((X, np.zeros(shape=(input_length, X.shape[1])))), n_neighbors=n_neighbors,
            metric="euclidean",
            metric_kwds=None,
            random_state=1949,
            n_trees=min(64, 5 + int(round((X.shape[0]) ** 0.5 / 20.0))),
            n_iters=max(5, int(round(np.log2(X.shape[0])))),
            max_candidates=max_candidates,
            low_memory=False,
            n_jobs=n_jobs,
            verbose=verbose
        )

        # Store the parameters
        self.input_length = input_length
        self.training_rows = X.shape[0]
        self.X_train = X
        self.y_train = y
        self.y_full = np.vstack((y, -1 * np.ones(shape=(input_length, 1))))
        self.n_neighbors = n_neighbors
        self.umap_kwargs = umap_kwargs
        self.rfc_kwargs = rfc_kwargs
        self.num_points = 0
        self.verbose = verbose
        self.n_jobs = n_jobs

    def predict(self, X_input):
        # Add the data
        self.index.update(xs_updated=X_input, updated_indices=range(self.training_rows, self.training_rows + self.input_length))
        X = np.vstack((self.X_train, X_input))

        # Perform the UMAP
        umap = UMAP(
            precomputed_knn=(*self.index.neighbor_graph, self.index),
            n_neighbors=self.n_neighbors,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            **self.umap_kwargs
        )
        res =  umap.fit_transform(X, self.y_full)
        print("UMAP finished.")
        X_train_mapped = res[:self.training_rows]
        X_umap = res[self.training_rows:]

        # Perform random forest classification
        rfc = RandomForestClassifier(n_jobs=self.n_jobs, **self.rfc_kwargs)
        rfc.fit(X_train_mapped, self.y_train.ravel())
        print("RFC finished.")

        # Do the final classification
        return rfc.predict(X_umap)

    def score(self, X_input, y_input, metric=v_measure_score):
        res = self.predict(X_input)
        print("Computing score...")
        # Compute the score
        return metric(y_input.ravel(), res)


if __name__ == "__main__":
    import clustering_figures as cf

    # Read in the datasets
    data_train = cf.to_xy_numeric(*cf.generate_sample(1., random_state=184))
    X_valid, y_valid = cf.to_xy_numeric(*cf.generate_sample(0.05, set_name="validate_patients", random_state=453))
    X_test, y_test = cf.to_xy_numeric(*cf.generate_sample(0.05, set_name="test_patients", random_state=453))
    #
    # # Do the training
    # test_size = min(X_valid.shape[0], X_test.shape[0])
    # params = ParameterGrid({
    #     "n_neighbors": [70, 110],
    #     "max_candidates": [20, 60],
    #     "umap__n_components": [4, 6, 8],
    #     "umap__target_weight": [.5, .8],
    #     "rfc__max_depth": [5, 7, 9],
    #     "rfc__max_features": ["sqrt", .5],
    #     "rfc__n_estimators": [65]
    # })
    #
    # best = None
    # score = None
    # results = []
    # try:
    #     for e in tqdm(params):
    #         print(f"Beginning {e}")
    #         umc = UMAPClassifier(
    #             *data_train,
    #             n_neighbors=e["n_neighbors"],
    #             input_length=test_size,
    #             max_candidates=e["max_candidates"],
    #             umap_kwargs={
    #                 "n_components": e["umap__n_components"],
    #                 "init": "pca",
    #                 "n_epochs": 10,
    #                 "target_weight": e["umap__target_weight"]
    #             },
    #             rfc_kwargs={
    #                 "max_depth": e["rfc__max_depth"],
    #                 "max_features": e["rfc__max_features"],
    #                 "min_samples_leaf": 10,
    #                 "n_estimators": e["rfc__n_estimators"]
    #             }
    #         )
    #
    #         print("Computing validation score...")
    #         start_time = time.perf_counter()
    #         score = umc.score(X_valid[:test_size], y_valid[:test_size], metric=accuracy_score)
    #         results.append((e, time.perf_counter() - start_time, score))
    # except KeyboardInterrupt as e:
    #     pass
    # finally:
    #     print("Results Found:")
    #     print(results)

    # Do the training
    test_size = min(X_valid.shape[0], X_test.shape[0])
    umc = UMAPClassifier(
        *data_train,
        n_neighbors=90,
        input_length=test_size,
        max_candidates=40,
        verbose=True,
        umap_kwargs={"n_components": 8, "init": "pca", "n_epochs": 30, "target_weight": .7},
        rfc_kwargs={"max_depth": 7, "max_features": .5, "min_samples_leaf": 10, "n_estimators": 60}
    )

    # Compute the score
    print("Now computing score on validation set...")
    start_time = time.perf_counter()
    print("Score:", umc.score(X_valid[:test_size], y_valid[:test_size], metric=accuracy_score))
    print("Total time: ", time.perf_counter() - start_time)
