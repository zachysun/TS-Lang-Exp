"""
Modified from source code:
https://github.com/daochenzha/ltsm/blob/main/prompt_bank/stat-prompt/tsfel/feature_extraction/features.py
"""
import numpy as np


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


if __name__ == "__main__":
    x = np.random.randn(10)
    print('signal:', x)
    print('Number of negative turning: ', negative_turning(x))
    print('Number of positive turning: ', positive_turning(x))
