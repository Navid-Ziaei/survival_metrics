import numpy as np


def get_ci_score(prediction,
                 survival_time,
                 death,
                 time):
    """
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky --> sooner death)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
    """

    N = len(prediction)
    A = np.zeros((N, N))  # acceptant pairs
    Q = np.zeros((N, N))
    tied_pairs = np.zeros((N, N))
    N_t = np.zeros((N, N))
    for i in range(N):
        A[i, np.where(survival_time[i] < survival_time)] = 1
        Q[i, np.where(prediction[i] > prediction)] = 1
        tied_pairs[i, np.where(prediction[i] == prediction)] = 1
        tied_pairs[i, prediction == prediction[i]] = 1

        if survival_time[i] <= time and 1 == death[i]:
            N_t[i, :] = 1

    Num = np.sum((A * N_t) * Q) + 0.5 * np.sum((A * N_t) * tied_pairs)
    Den = np.sum(A * N_t)

    if 0 == Num and 0 == Den:
        result = -1  # not able to compute c-index!
    else:
        result = float(Num / Den)
    return result


def get_brier_score(prediction,
                    survival_time,
                    death,
                    time):
    """
    A function for calculating the Brier score for uncensored patients
    :param prediction: output of the model
    :param survival_time: the ground truth
    :param death: ground truth of the event
    :param time: time of evaluation (time-horizon when evaluating Brier score)
    :return: the brier score
    """

    y_true = (np.transpose(survival_time <= time) * death).astype(float)
    num = np.sum(y_true * np.square(prediction - y_true))
    den = np.maximum(1., np.sum(y_true))
    return num / den
