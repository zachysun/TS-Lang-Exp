"""
Modified from source code:
https://github.com/daochenzha/ltsm/blob/main/prompt_bank/stat-prompt/tsfel/feature_extraction/
"""
import numpy as np

feature_extraction_functions = []


def feature_extraction_function(func):
    feature_extraction_functions.append(func)
    return func


@feature_extraction_function
def autocorr(signal):
    """Computes autocorrelation of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which autocorrelation is computed

    Returns
    -------
    float
        Cross correlation of 1-dimensional sequence

    """
    signal = np.array(signal)
    return float(np.correlate(signal, signal))


@feature_extraction_function
def negative_turning(signal):
    """Computes number of negative turning points of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which negative turning points are counted
    Returns
    -------
    float
        Number of negative turning points

    """
    diff_sig = np.diff(signal)
    array_signal = np.arange(len(diff_sig[:-1]))
    negative_turning_pts = np.where((diff_sig[array_signal] < 0) & (diff_sig[array_signal + 1] > 0))[0]

    return len(negative_turning_pts)


@feature_extraction_function
def positive_turning(signal):
    """Computes number of positive turning points of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which positive turning points are counted

    Returns
    -------
    float
        Number of positive turning points

    """
    diff_sig = np.diff(signal)
    array_signal = np.arange(len(diff_sig[:-1]))
    positive_turning_pts = np.where((diff_sig[array_signal + 1] < 0) & (diff_sig[array_signal] > 0))[0]

    return len(positive_turning_pts)


@feature_extraction_function
def mean_abs_diff(signal):
    """Computes mean absolute differences of the signal.

   Feature computational cost: 1

   Parameters
   ----------
   signal : nd-array
       Input from which mean absolute deviation is computed

   Returns
   -------
   float
       Mean absolute difference result

   """
    return np.mean(np.abs(np.diff(signal)))


@feature_extraction_function
def mean_diff(signal):
    """Computes mean of differences of the signal.

   Feature computational cost: 1

   Parameters
   ----------
   signal : nd-array
       Input from which mean of differences is computed

   Returns
   -------
   float
       Mean difference result

   """
    return np.mean(np.diff(signal))


@feature_extraction_function
def median_abs_diff(signal):
    """Computes median absolute differences of the signal.

   Feature computational cost: 1

   Parameters
   ----------
   signal : nd-array
       Input from which median absolute difference is computed

   Returns
   -------
   float
       Median absolute difference result

   """
    return np.median(np.abs(np.diff(signal)))


@feature_extraction_function
def median_diff(signal):
    """Computes median of differences of the signal.

   Feature computational cost: 1

   Parameters
   ----------
   signal : nd-array
       Input from which median of differences is computed

   Returns
   -------
   float
       Median difference result

   """
    return np.median(np.diff(signal))
