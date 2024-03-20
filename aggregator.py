import numpy as np
import os.path
from multiprocessing import Pool
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy import stats
from tqdm import tqdm

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
    ("zeros", 15)
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
IN_FILE_NIGHT_NUM = "nights"
IN_FILE_SAMPLE_FREQUENCIES = [100, 1]
IN_FILE_LABELS_LIST = "labels"
# Train-test split ratios
TRAIN_PORTION = 0.5
TEST_PORTION = 0.3
VALIDATE_PORTION = 0.2
# -------------------------------------------------------------------
assert TRAIN_PORTION + TEST_PORTION + VALIDATE_PORTION == 1.0


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

    # This is just here for testing.
    @staticmethod
    def zeros(mats: list[np.ndarray] | None, _: list[str], options: list):
        # Read in the options
        dim, = options

        # Check if we're retrieving the post-processing labels
        if mats is None:
            return [f"zero-{i}" for i in range(dim)]

        # Return zeros of the right dimension
        return np.zeros(dim)


# Set up the processors
_processors = [(getattr(Processor, details[0]), details[1:]) for details in PROCESSORS]
labels = None


# -------------------------------------------------------------------
# Main Aggregator
# This section defines a class
# -------------------------------------------------------------------
class Aggregator:
    def __init__(self):
        global labels
        # Load in the npz file
        self.in_file = np.load(IN_FILE_NAME)

        # Determine the labels for each frequency, and the patients
        labels = [self.in_file[f"{IN_FILE_LABELS_LIST}-{freq}hz"] for freq in IN_FILE_SAMPLE_FREQUENCIES]
        self.patients = self.in_file[IN_FILE_KEYS_LIST]
        self.num_nights = self.in_file[IN_FILE_NIGHT_NUM]

    def process_all(self, n_proc=24):
        # Determine the new output structure
        labels = []
        for func, opts in _processors:
            labels.append(func(None, labels, opts))

        # Create a new directory, if it does not exist, and save all the files out
        save = not os.path.exists("./tmp")
        if save:
            Path("./tmp").mkdir()

        # Load in every file and save out the ones that are not present
        files = []
        for id in tqdm(self.patients):
            for night in range(self.num_nights):
                try:
                    if save:
                        for freq in IN_FILE_SAMPLE_FREQUENCIES:
                            name = f"{id}{night + 1}-{freq}HZ"
                            np.save(f"./tmp/{name}", self.in_file[name])
                    files.append(f"{id}{night + 1}")
                except KeyError as e:
                    print(e)

        # Loop through every patient and convert them to the new format
        with Pool(n_proc) as p:
            result = []
            for r in tqdm(p.imap(self.process_person, files), total=len(files)):
                result.append(r)

        # Create the output dictionary
        out = dict()
        out["all_patients"] = self.patients
        out["labels"] = labels
        out["num_nights"] = self.num_nights

        # Save all the people who were successfully converted
        for file, res in zip(files, result):
            if res is None:
                continue
            out[file] = res

        # Make a train-test-validate split
        patients_tv, patients_test = train_test_split(self.patients, test_size=TEST_PORTION)
        patients_train, patients_validate = train_test_split(patients_tv, test_size=(
                VALIDATE_PORTION / (VALIDATE_PORTION + TRAIN_PORTION))
        )
        out["test_patients"] = patients_test
        out["train_patients"] = patients_train
        out["validate_patients"] = patients_validate

        # Save the new file
        np.savez(OUT_FILE_NAME, **out)

    @staticmethod
    def process_person(file):
        try:
            # Get the arrays for each frequency
            mats = [np.load(f"./tmp/{file}-{freq}HZ.npy") for freq in IN_FILE_SAMPLE_FREQUENCIES]

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
                for func, opts in _processors:
                    vectors.append(func(mats_seg, labels, opts))
                entry_out.append(np.concatenate(vectors))

            # Return the aggregated data, as an array
            return np.array(entry_out)
        except FileNotFoundError:
            return None


if __name__ == "__main__":
    agg = Aggregator()
    agg.process_all()

    # file = np.load("sleep-cassette-aggregate.npz")
    # print(list(file.keys()))
    # print(file["labels"])
    # print(file["patients"])
    # print(file["SC4711"])
    # print(file["patients"])
    # print(file["labels-100hz"])
    # print(file["labels-1hz"])
