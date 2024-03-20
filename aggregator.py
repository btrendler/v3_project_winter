import numpy as np
from scipy import stats

# -------------------------------------------------------------------
# Configuration Options:
# -------------------------------------------------------------------
# Number of seconds per data block
BLOCK_LENGTH = 10.
# The processor to use for each data block. Valid keys are:
# - fourier: Do the fourier transform on the data. Options are a tuple, containing the list of columns to apply to and
#     the number of peak wavelengths to identify
# - mean: Average this column across the interval. Options are a list of columns to apply to
# - mode: Take the most common value from this column as the value across the whole interval
# Additional processors can be programmed in the next section.
# Note: every column MUST have an aggregator specified, or it will not be passed through to the final data.
PROCESSORS = [
    ("zeros", ["EEG FPZ-CZ"])
    # ("fourier", ["EEG FPZ-CZ", "EEG PZ-OZ", "EOG HORIZONTAL", "EMG SUBMENTAL"], 5),
    # ("mean", ["RESP ORO-NASAL", "TEMP RECTAL"]),
    # ("mode", ["_HYPNO"])
]
# The label for the dependent variable
DEPENDENT_LABEL = "_HYPNO"
# How to handle data blocks where the dependent variable takes multiple values; valid options are 'withhold', 'ignore'
DEPENDENT_CHANGE_METHOD = "withhold"
# Input and output file names
OUT_FILE_NAME = "sleep-cassette-aggregate.npz"
IN_FILE_NAME = "sleep-cassette.npz"
# Labels for entries in the npz file
IN_FILE_KEYS_LIST = "patients"
IN_FILE_SAMPLE_FREQUENCIES = [100, 1]
IN_FILE_LABELS_LIST = ["labels"]


# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------
def find_tuple_index(array: list[list[str]], value: str):
    for i in range(len(array)):
        try:
            return i, array[i].index(value)
        except ValueError:
            continue
    raise ValueError("Entry not found in 2D string list")


# -------------------------------------------------------------------
# Data Processor
# This section defines a class containing all the processors. Each processor is a function with the following signature:
# """
# A data aggregator.
# :param mat: The list of matrices to process, or none if the overarching aggregator is requesting labels for values
# :param labels: The list of lists of column labels - the lists in the outer array correspond to the frequencies
# :param options: The options with which to run this aggregator - first option is always which columns to apply to.
# :return: A tuple containing the found values, or labels for the found values if mat is None
# """
#
# -------------------------------------------------------------------
class Processor:
    @staticmethod
    def fourier(mats: list[np.ndarray] | None, labels: list[list[str]], options: list):
        # TODO
        ...

    @staticmethod
    def mean(mats: list[np.ndarray] | None, labels: list[list[str]], options: list):
        # TODO
        ...

    @staticmethod
    def mode(mats: list[np.ndarray] | None, labels: list[list[str]], options: list):
        # Read in the options
        targets, = options

        # Check if we're retrieving the post-processing labels
        if mats is None:
            return [target + "-Mean" for target in targets]

        # Process the targets, in order - this order MUST NOT CHANGE between the post-processing labels being returned,
        #  and the values being computed.
        out = []
        for target in targets:
            # Get the frequency index & column index
            freq_idx, label_col = find_tuple_index(labels, target)
            # Append the mode
            out.append(stats.mode(mats[freq_idx][:, label_col])[0])

        # Return the found values. Can be a list or an ndarray.
        return out

    @staticmethod
    def zeros(mats: list[np.ndarray] | None, labels: list[str], options: list):
        return np.zeros(3)


# -------------------------------------------------------------------
# Main Aggregator
# This section defines a class
# -------------------------------------------------------------------
class Aggregator:
    def __init__(self):
        # Load in the npz file
        self.in_file = np.load(IN_FILE_NAME)
        self._processors = [(getattr(Processor, name), options) for name, options in PROCESSORS]

        # Determine the labels for each frequency
        self.labels = None

    def process_all(self, n_proc=24):
        self.process_person("SC4801")

    def process_person(self, id):
        # Get the arrays for each frequency
        mats = [self.in_file[f"{id}-{freq}HZ"] for freq in IN_FILE_SAMPLE_FREQUENCIES]

        # Determine the number of seconds in this sample & the segment divisions
        seconds = max(mat.shape[0] / freq for mat, freq in zip(mats, IN_FILE_SAMPLE_FREQUENCIES))
        segments = np.arange(0., seconds, BLOCK_LENGTH)

        # Loop through the segments
        entry_out = []
        for start in segments:
            # If there isn't enough data left, return
            if start + BLOCK_LENGTH > seconds:
                break
            # Get the segments
            mats_seg = []
            for mat, freq in zip(mats, IN_FILE_SAMPLE_FREQUENCIES):
                start_index = int(freq * start)
                end_index = int(start_index + freq * BLOCK_LENGTH)
                mats_seg.append(mat[start_index:end_index, :])

            # Call each processor on the found segments
            vectors = []
            for func, opts in self._processors:
                vectors.append(func(mats_seg, self.labels, *opts[1:]))
            entry_out.append(np.concatenate(vectors))

        # Return the aggregated data, as an array
        return np.array(entry_out)


if __name__ == "__main__":
    agg = Aggregator()
    agg.process_all()
    # file = np.load("sleep-cassette.npz")
    # print(list(file.keys()))
    # print(file["labels-100hz"])
    # print(file["labels-1hz"])
