import numpy as np
import glob
from itertools import zip_longest


class SplitNPZ:
    def __init__(self, filename_pattern):
        # List all the files matching the pattern
        self.npzs = [np.load(f) for f in glob.glob(filename_pattern)]

    def __getitem__(self, item):
        for npz in self.npzs:
            if item in npz:
                return npz[item]
        raise KeyError(f"{item} is not a file in the multi-archive")

    def keys(self):
        return [item for npz in self.npzs for item in npz.keys()]

    def __contains__(self, item):
        for npz in self.npzs:
            if item in npz:
                return True
        return False

    @staticmethod
    def save(filename, per_file=0.3, **kwargs):
        # Determine the number per file
        file_size = int(len(kwargs) * per_file) if type(per_file) is float else per_file
        # Group into each of the save files, and write out
        for i, batch in enumerate(zip_longest(*([iter(kwargs.items())] * file_size))):
            np.savez(filename(i), **dict(filter(lambda x: x is not None, batch)))
