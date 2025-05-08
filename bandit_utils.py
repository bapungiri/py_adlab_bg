import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d


def from_csv(fp):
    """This function primarily written to handle data from anirudh's bandit task/processed data

    Parameters
    ----------
    fp : .csv file name
        File path to the csv file that contains the data

    Returns
    -------
    _type_
        _description_
    """
    df = pd.read_csv(fp)
    port1_prob = df.loc[:, "rewprobfull1"]
    port2_prob = df.loc[:, "rewpcrobfull2"]
    prob_choice = df.loc[:, "rw"].to_numpy()
    is_choice_high = np.max(np.vstack((port1_prob, port2_prob)), axis=0) == prob_choice

    # whr_reward_trial = np.where(is_reward == 1)[0]
    # rwd_mov_avg = np.convolve(is_reward, np.ones(150) / 150, mode="same")
    # rwd_mov_avg_smth = gaussian_filter1d(rwd_mov_avg, 20)
    # rwd_prob_corr = stats.pearsonr(port1_prob, port2_prob)
    # date_time = pd.to_datetime(df["eptime"].to_numpy(), unit="s")
    df["is_choice_high"] = is_choice_high
    return df


def choices_to_strseq_Beron2022(switch_prob, seq):
    """Beron et al. 2022 coding of actions sorted by most recent outcome"""
    # Coding seq of 1,-1,2,-2 to A,a,B,b to be compatible with Beron 2022
    seq = seq.astype(str)
    seq[seq == "-1"] = "a"
    seq[seq == "1"] = "A"
    seq[seq == "-2"] = "b"
    seq[seq == "2"] = "B"

    # Sorting index by most recent reward, then by second most recent reward
    sort_indx = np.lexsort((seq[:, 1], seq[:, 2]))

    # joining to seq into continuous string
    seq = np.array(["".join(map(str, _)) for _ in seq])

    return switch_prob[sort_indx], seq[sort_indx]
