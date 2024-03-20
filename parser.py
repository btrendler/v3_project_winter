"""
This file serves to convert bulk edf files to one npz file, with the following named arrays:
    - labels: A 1-D Numpy string array containing the column labels
    - patients: A 1-D Numpy string array listing every patient
    - nights: An int, the number of nights
    - <patient data file>: A 2-D Numpy array containing the
NOTE: Not all combinations of patient/nights are guaranteed to exist by this parser.
"""

import numpy as np
import os
import pyedflib
import re
from itertools import groupby
from tqdm import tqdm


###############################################################
#  Utility functions for edf -> npz conversion.
###############################################################

_HYPNOGRAM_LABEL = "_HYPNO"
_HYPNOGRAM_STATES = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
    "Sleep stage ?": np.nan
}

# Utility function to extract patient & night information
_patient_and_night = lambda k:k[:6]

# Using this on the dataset yields
#   {'EVENT MARKER', 'TEMP RECTAL', 'RESP ORO-NASAL', 'EEG FPZ-CZ', 'EOG HORIZONTAL', 'EEG PZ-OZ', 'EMG SUBMENTAL'}
#   100 as the lcm
def _get_all_labels(data_folder: str) -> dict:
    """
    Get a list containing every data label contained in any of the files in the given folder
    :param data_folder: The folder path
    :return: A set containing all the data labels
    """
    # List all the files in the directory, grouped by the
    data_files = os.listdir(data_folder)
    data_files.sort()

    # Loop through all the participant/night combos, and open both files
    labels = dict()
    for pid, files in groupby(data_files, _patient_and_night):
        # Get the PSG and Hypnogram files
        l = list(files)
        l.sort()
        psg, hyp = l

        # Load the PSG file and read out the label titles
        psg = pyedflib.EdfReader(data_folder + "/" + psg)
        for hdr in psg.getSignalHeaders():
            rate = int(hdr['sample_rate'])
            if rate not in labels:
                labels[rate] = set()
            labels[rate].add(hdr['label'].upper())

        # Close the file
        psg.close()

    # Return the found labels, and the LCM
    return labels


# Utility function to load all the EDF files in a directory
def _import_all_edfs(data_folder: str, signals: dict[int, list]) -> dict[str, np.ndarray]:
    """
    Imports every EDF file in the given directory, using the str -> int label mapping defined by the signals list
    :param data_folder: The folder path containing the EDFs
    :param signals: The list of signals to load
    :return: A dictionary mapping patient + night + frequency to a numpy array
    """

    # List all the files in the directory, grouped by the
    data_files = os.listdir(data_folder)
    data_files.sort()

    # Loop through all the participant/night combos, and open both files
    data = dict()
    print("Importing all EDFs...")

    def _task(pair):
        l = list(pair)
        l.sort()
        psg, hyp = l
        psg = data_folder + "/" + psg
        hyp = data_folder + "/" + hyp

        return _import_edf_combo(psg, hyp, signals)

    #from multiprocessing import Pool
    #with Pool(8) as p:
    #    res = p.map(_task, list(groupby(data_files, _patient_and_night)))
    #    for imported in res:
    #        for freq, arr in imported:
    #            data[pid + "-" + str(freq) + "HZ"] = arr
    for pid, files in tqdm(groupby(data_files, _patient_and_night)):
        # Get the PSG and Hypnogram files
        l = list(files)
        l.sort()
        psg, hyp = l
        psg = data_folder + "/" + psg
        hyp = data_folder + "/" + hyp

        # Import the arrays
        imported = _import_edf_combo(psg, hyp, signals)
        for freq, arr in imported:
            data[pid + "-" + str(freq) + "HZ"] = arr

    # Return the dictionary
    return data


def _import_edf_combo(psg_file: str, hyp_file: str, signals: dict[int, list[str]]) -> list[tuple[int, np.ndarray]]:
    """
    Import the data contained in two EDF files - one for the PSG, one for the Hypnograph
    :param psg_file: The PSG EDF file
    :param hyp_file: The Hypnograph EDF file
    :param signals: The signals to import (annotations are imported even if not specified here)
    :return: A list containing tuples mapping frequency to a numpy array
    """
    print("Loading file...", psg_file)

    # First, open the two files
    psg = pyedflib.EdfReader(psg_file)
    hyp = pyedflib.EdfReader(hyp_file)

    # Loop through the frequencies and add them to the output
    output = list()
    for freq in signals:
        # Get the labels with this frequency
        labels = signals[freq]

        # Create the numpy array, with freq * file_seconds rows, and the correct number of columns
        freq_output = np.zeros((psg.getFileDuration() * freq, len(labels))) * np.NaN

        # Loop through the signals and read the data into the array
        for i, s in enumerate(psg.getSignalLabels()):
            # Check if this signal is in the current frequency
            name = s.upper()
            if name not in signals[freq]:
                continue

            # Read that signal into the corresponding column
            freq_output[:,labels.index(name)] = psg.readSignal(i)

        # Read from the hypnogram, if necessary (we treat it as frequency-one)
        if freq == 1:
            # Read the annotations
            times, durations, titles = hyp.readAnnotations()
            times = list(times.astype(int))

            # Create a time-series-format column for the 1-hz array.
            state = np.nan
            hypnogram = np.zeros(freq_output.shape[0]) * np.nan
            for i in range(hypnogram.shape[0]):
                if i in times:
                    title = titles[times.index(i)]
                    if title in _HYPNOGRAM_STATES:
                        state = _HYPNOGRAM_STATES[title]
                hypnogram[i] = state

            freq_output[:,labels.index(_HYPNOGRAM_LABEL)] = hypnogram

        # Add to the output list
        output.append((freq, freq_output))

    # Close the files
    psg.close()
    hyp.close()

    # Return the output
    return output


if __name__ == "__main__":
    """
        When run with no parameters, this program converts all the edf files to one npz file.
    """
    data_folder = "data/sleep-cassette"
    out_folder = "sleep-cassette.npz"
    name_length = 5
    """"""

    # Find the information about the cassette data
    labels = _get_all_labels(data_folder)
    print("Labels found:", labels)
    labels[1].add(_HYPNOGRAM_LABEL)

    # Remove the 'EVENT MARKER' since we don't know what it does, put in alphabetical order
    labels_named = dict()
    for key in labels:
        # Convert to a list
        value = labels[key]
        value.discard('EVENT MARKER')
        value = list(value)
        value.sort()

        # Store it in the new dictionary
        new_key = "labels-" + str(key) + "hz"
        labels[key] = value
        labels_named[new_key] = value

    # Convert each patient into a numpy array
    patient_arrays = _import_all_edfs(data_folder, signals=labels)

    # Determine the set of all patients, and the number of nights
    patients = list(set(key[:name_length] for key in patient_arrays))
    night_pattern = re.compile(r'\d+')
    num_nights = len(set(night_pattern.search(key[name_length:]).group() for key in patient_arrays))

    # Write out the final npz file
    np.savez_compressed(out_folder, patients=patients, nights=num_nights, **patient_arrays, **labels_named)
