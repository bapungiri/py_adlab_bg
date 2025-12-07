import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import os
# import re
from functools import partial
import datetime

# import openpyxl as xl
# from scipy.ndimage import uniform_filter
# from scipy.stats import sem
# from numpy.linalg import norm
import seaborn as sns

plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
sns.set_theme()
sns.set_style("ticks")


def get_files(
    log_folder="C:/Users/dlab/Downloads/AutoTrainerModular/New folder/Brat",
    extension=".dat",
):
    """gets all the files from input and checks if they are csv"""
    import os

    files = []
    if type(log_folder) == list:
        for fol in log_folder:
            os.chdir(os.path.realpath(fol))
            for f in os.listdir(os.path.realpath(fol)):
                if f.endswith(extension):
                    files.append(f)
    else:
        for f in os.listdir(os.path.realpath(log_folder)):
            os.chdir(os.path.realpath(log_folder))
            if f.endswith(extension):
                files.append(f)
    return files


# def merge_files_to_df(files):
#     df = pd.concat(map(partial(pd.read_csv, header=None), files), ignore_index = True).dropna().astype(int)
#     return df
def merge_files_to_df(files):
    names = ["event", "value", "sm", "tp", "smtime", "eptime", "dw", "ww"]
    df = (
        pd.concat(
            map(partial(pd.read_csv, header=None, names=names), files),
            ignore_index=True,
        )
        .dropna()
        .astype(int)
    )
    return df


def trializer_v2(
    df,
    stateMachines,
    trialStartMarker,
    trialHitMarker,
    trialMissMarker,
    sessionStartMarker,
):  # ,trialProbMarker):
    """Makes a trialwise df of .dat files,
    Input = df of .dat, List of stateMachines to be analyzed, trialStartMarker, trialHitMarker, trialMissMarker, sessionStartMarker
    Output = trialwise df
    """
    lateSess = df[df[2].isin(stateMachines)]
    lateSessNPtimes = lateSess[lateSess[0] == trialStartMarker][4]
    lateSessRewtimes = lateSess[lateSess[0] == trialHitMarker][4]
    lateSessMisstimes = lateSess[lateSess[0] == trialMissMarker][4]

    print(
        " no of trials in SMs = ",
        lateSessNPtimes.shape,
        "\n",
        "no of rewards in SMs = ",
        lateSessRewtimes.shape,
        "\n",
        "no of misses in SMs = ",
        lateSessMisstimes.shape,
        "\n",
        "rewards + misses = ",
        lateSessRewtimes.shape[0] + lateSessMisstimes.shape[0],
        "\n",
    )

    pd.options.mode.chained_assignment = None
    lateSessdf = pd.DataFrame()
    lateSessdf["trial#"] = np.arange(0, len(lateSessNPtimes))
    lateSessdf["trialstart"] = lateSessNPtimes.values
    lateSessdf["port"] = lateSess[lateSess[0] == trialStartMarker][
        1
    ].values  # what other marker can be used for trial choice? 20 and 21, but those are over-reported.
    lateSessdf["reward"] = np.nan
    lateSessdf["trialend"] = np.nan
    lateSessdf["session#"] = np.nan
    lateSessdf["rewprob"] = np.nan
    sessioncounter = 0
    npcounter = 0

    event_lookup = {
        trialHitMarker: 1,
        trialMissMarker: 0,
    }

    for ind, event in enumerate(lateSess[0]):
        if event == trialStartMarker:
            nextEvents = lateSess.iloc[ind:]
            try:
                statusIndex = min(
                    np.where(nextEvents[0].isin([trialMissMarker, trialHitMarker]))[0]
                )
                # rewprob = min(np.where(nextEvents[0] == trialProbMarker)[0])
                lateSessdf.loc[npcounter, "reward"] = event_lookup[
                    nextEvents.iloc[statusIndex][0]
                ]
                lateSessdf.loc[npcounter, "trialend"] = nextEvents.iloc[statusIndex, 4]
                # lateSessdf.loc[npcounter, 'rewprob'] = nextEvents.iloc[rewprob, 1]
            except ValueError:
                lateSessdf.loc[npcounter, "reward"] = np.nan
                lateSessdf.loc[npcounter, "trialend"] = np.nan
                # lateSessdf.loc[npcounter, 'rewprob'] = np.nan
            npcounter += 1
        elif event == sessionStartMarker:
            lateSessdf.loc[npcounter:, "session#"] = sessioncounter
            sessioncounter += 1

    pd.options.mode.chained_assignment = "warn"
    return lateSessdf


def trializer_v3(
    df,
    list_sessionMarker,
    trialStartMarker,
    trialHitMarker,
    trialMissMarker,
    sessionStartMarker,
    rewardProbMarker1,
    rewardProbMarker2,
    animal,
    arms=2,
):
    """Function to convert a df of joined .dat files with column names in order :
    'event', 'value', 'sm', 'tp', 'smtime', 'eptime', 'dw', 'ww'

    INPUTS: df, list of session state machine numbers, trial start marker, trial rewarded marker,
    trial miss/unrewarded marker, session start marker, trial probability marker, session reward prob marker,
    animal name, num arms in task

    OUTPUT: single trial-wise dataframe of one animal
    2023-10-26
    """
    # initialize var
    sessnum = 0
    npcounter = 0
    sess_list = []
    rewprobl = 0
    rewprobl2 = 0
    event_lookup = {
        trialHitMarker: 1,
        trialMissMarker: 0,
    }
    list_events = [
        trialStartMarker,
        trialHitMarker,
        trialMissMarker,
        sessionStartMarker,
        rewardProbMarker1,
        rewardProbMarker2,
    ]

    # filter events as np array to increase speed and avoid repetitive indexing
    filtered_events = df[
        (df["sm"].isin(list_sessionMarker)) & (df["event"].isin(list_events))
    ].to_numpy()
    indices = np.where(
        (filtered_events[:, 0] == trialStartMarker)
        | (filtered_events[:, 0] == sessionStartMarker)
        | (filtered_events[:, 0] == rewardProbMarker1)
        | (filtered_events[:, 0] == rewardProbMarker2)
    )[
        0
    ]  # indices of events

    for ind in indices:
        # extract information from the current trialStartMarker
        trialstart = filtered_events[ind, 4]
        port = filtered_events[ind, 1]  # needs to change
        datetime = filtered_events[ind, 5]
        task = filtered_events[ind, 2]
        event = filtered_events[ind, 0]

        # select events occurring after trialStartMarker
        nextEvents = filtered_events[ind:]

        try:
            # find the index of trialMissMarker or trialHitMarker in nextEvents
            status_indices = np.where(
                (nextEvents[:, 0] == trialMissMarker)
                | (nextEvents[:, 0] == trialHitMarker)
            )[0]
            statusIndex = status_indices[0] if status_indices.size > 0 else None

            # find the index of trialProbMarker in nextEvents

            # contLine: if np.any(nextEvents[:, 0] == trialProbMarker) else None

            if statusIndex is not None:
                # find reward, trialend, and rewprob based on identified events
                reward = event_lookup[nextEvents[statusIndex, 0]]
                trialend = nextEvents[statusIndex, 4]
                # rewprob = nextEvents[rewprob_index, 1] if rewprob_index is not None else np.nan
            else:
                # if trialMissMarker or trialHitMarker not found, nan
                reward = np.nan
                trialend = np.nan
                # rewprob = np.nan
        except IndexError:
            # set values to nan in case of issues
            reward = np.nan
            trialend = np.nan
            # rewprob = np.nan

        # append to sess_list
        sess_list.append(
            [
                npcounter,
                trialstart,
                port,
                reward,
                trialend,
                sessnum,
                datetime,
                task,
                event,
                rewprobl,
                rewprobl2,
            ]
        )
        npcounter += 1

        if filtered_events[ind, 0] == sessionStartMarker:
            # update session number when sessionStartMarker found
            rewprobl = 0
            rewprobl2 = 0
            sessnum += 1

        if filtered_events[ind, 0] == rewardProbMarker1:
            rewprobl = filtered_events[ind, 4]

        if filtered_events[ind, 0] == rewardProbMarker2:
            rewprobl2 = filtered_events[ind, 4]

    # create pandas dataframe from the list of all session data
    sessdf = pd.DataFrame(
        sess_list,
        columns=[
            "trial#",
            "trialstart",
            "port",
            "reward",
            "trialend",
            "session#",
            "eptime",
            "task",
            "event",
            "rewprobfull1",
            "rewprobfull2",
        ],
    )

    # remove all lines corresponding to start of a session or reward prob marker
    sessdf = (
        sessdf[
            (sessdf.event != sessionStartMarker)
            & (sessdf.event != rewardProbMarker1)
            & (sessdf.event != rewardProbMarker2)
        ]
        .reset_index(drop=True)
        .drop(columns="event")
    )

    # remove all lines that have (trialend-trialstart)> 5500 or negative (sum of tone + wait time)
    # is this check required?

    # reset trial numbers
    sessdf["trial#"] = np.arange(len(sessdf))

    # add animal information
    sessdf["animal"] = animal

    # add datetime
    sessdf["datetime"] = pd.to_datetime(sessdf.eptime, unit="s")

    return sessdf


def trializer_v3_ind(
    df,
    list_sessionMarker,
    trialStartMarker,
    trialHitMarker,
    trialMissMarker,
    sessionStartMarker,
    rewardProbMarker1,
    rewardProbMarker2,
    animal,
    arms=2,
):
    """Function to convert a df of joined .dat files with column names in order :
    'event', 'value', 'sm', 'tp', 'smtime', 'eptime', 'dw', 'ww'

    INPUTS: df, list of session state machine numbers, trial start marker, trial rewarded marker,
    trial miss/unrewarded marker, session start marker, trial probability marker, session reward prob marker,
    animal name, num arms in task

    OUTPUT: single trial-wise dataframe of one animal
    2023-10-26
    """
    # initialize var
    sessnum = 0
    npcounter = 0
    sess_list = []
    rewprobl = 0
    rewprobl2 = 0
    event_lookup = {
        trialHitMarker: 1,
        trialMissMarker: 0,
    }
    list_events = [
        trialStartMarker,
        trialHitMarker,
        trialMissMarker,
        sessionStartMarker,
        rewardProbMarker1,
        rewardProbMarker2,
    ]

    # filter events as np array to increase speed and avoid repetitive indexing
    filtered_events = df[
        (df["sm"].isin(list_sessionMarker)) & (df["event"].isin(list_events))
    ].to_numpy()
    indices = np.where(
        (filtered_events[:, 0] == trialStartMarker)
        | (filtered_events[:, 0] == sessionStartMarker)
        | (filtered_events[:, 0] == rewardProbMarker1)
        | (filtered_events[:, 0] == rewardProbMarker2)
    )[0]

    for ind in indices:
        # extract information from the current trialStartMarker
        trialstart = filtered_events[ind, 4]
        port = filtered_events[ind, 1]
        datetime = filtered_events[ind, 5]
        task = filtered_events[ind, 2]
        event = filtered_events[ind, 0]

        # select events occurring after trialStartMarker
        nextEvents = filtered_events[ind:]

        try:
            # find the index of trialMissMarker or trialHitMarker in nextEvents
            status_indices = np.where(
                (nextEvents[:, 0] == trialMissMarker)
                | (nextEvents[:, 0] == trialHitMarker)
            )[0]
            statusIndex = status_indices[0] if status_indices.size > 0 else None

            # find the index of trialProbMarker in nextEvents

            # contLine: if np.any(nextEvents[:, 0] == trialProbMarker) else None

            if statusIndex is not None:
                # find reward, trialend, and rewprob based on identified events
                reward = event_lookup[nextEvents[statusIndex, 0]]
                trialend = nextEvents[statusIndex, 4]
                # rewprob = nextEvents[rewprob_index, 1] if rewprob_index is not None else np.nan
            else:
                # if trialMissMarker or trialHitMarker not found, nan
                reward = np.nan
                trialend = np.nan
                # rewprob = np.nan
        except IndexError:
            # set values to nan in case of issues
            reward = np.nan
            trialend = np.nan
            # rewprob = np.nan

        # append to sess_list
        sess_list.append(
            [
                npcounter,
                trialstart,
                port,
                reward,
                trialend,
                sessnum,
                datetime,
                task,
                event,
                rewprobl,
                rewprobl2,
            ]
        )
        npcounter += 1

        if filtered_events[ind, 0] == sessionStartMarker:
            # update session number when sessionStartMarker found
            rewprobl = 0
            rewprobl2 = 0
            sessnum += 1

        if (filtered_events[ind, 0] == rewardProbMarker1) & (
            filtered_events[ind, 1] == 1
        ):
            rewprobl = filtered_events[ind, 4]

        if (filtered_events[ind, 0] == rewardProbMarker2) & (
            filtered_events[ind, 1] == 2
        ):
            rewprobl2 = filtered_events[ind, 4]

    # create pandas dataframe from the list of all session data
    sessdf = pd.DataFrame(
        sess_list,
        columns=[
            "trial#",
            "trialstart",
            "port",
            "reward",
            "trialend",
            "session#",
            "eptime",
            "task",
            "event",
            "rewprobfull1",
            "rewprobfull2",
        ],
    )

    # remove all lines corresponding to start of a session or reward prob marker
    sessdf = (
        sessdf[
            (sessdf.event != sessionStartMarker)
            & (sessdf.event != rewardProbMarker1)
            & (sessdf.event != rewardProbMarker2)
        ]
        .reset_index(drop=True)
        .drop(columns="event")
    )

    # remove all lines that have (trialend-trialstart)> 5500 or negative (sum of tone + wait time)
    # is this check required?

    # reset trial numbers
    sessdf["trial#"] = np.arange(len(sessdf))

    # add animal information
    sessdf["animal"] = animal

    # add datetime
    sessdf["datetime"] = pd.to_datetime(sessdf.eptime, unit="s")

    return sessdf


def trializer_v3_test(
    df,
    list_sessionMarker,
    trialStartMarker,
    trialHitMarker,
    trialMissMarker,
    sessionStartMarker,
    rewardProbMarker1,
    rewardProbMarker2,
    nosePokeMarker1,
    nosePokeMarker2,
    animal,
    arms=2,
):
    """Function to convert a df of joined .dat files with column names in order :
    'event', 'value', 'sm', 'tp', 'smtime', 'eptime', 'dw', 'ww'

    INPUTS: df, list of session state machine numbers, trial start marker, trial rewarded marker,
    trial miss/unrewarded marker, session start marker, trial probability marker, session reward prob marker,
    animal name, num arms in task

    OUTPUT: single trial-wise dataframe of one animal
    2023-10-26
    """
    # initialize var
    sessnum = 0
    npcounter = 0
    sess_list = []
    rewprobl = 0
    rewprobl2 = 0
    event_lookup = {
        trialHitMarker: 1,
        trialMissMarker: 0,
    }
    port_lookup = {
        nosePokeMarker1: 1,
        nosePokeMarker2: 2,
        trialMissMarker: np.nan,
        trialHitMarker: np.nan,
        83: np.nan,
    }
    list_events = [
        trialStartMarker,
        trialHitMarker,
        trialMissMarker,
        sessionStartMarker,
        rewardProbMarker1,
        rewardProbMarker2,
        nosePokeMarker1,
        nosePokeMarker2,
    ]

    # filter events as np array to increase speed and avoid repetitive indexing
    filtered_events = df[
        (df["sm"].isin(list_sessionMarker)) & (df["event"].isin(list_events))
    ].to_numpy()
    # print(filtered_events[0:50, :])
    indices = np.where(
        (filtered_events[:, 0] == trialStartMarker)
        | (filtered_events[:, 0] == sessionStartMarker)
        | (filtered_events[:, 0] == rewardProbMarker1)
        | (filtered_events[:, 0] == rewardProbMarker2)
    )[
        0
    ]  # indices of events

    for ind in indices:
        # extract information from the current trialStartMarker
        trialstart = filtered_events[ind, 4]
        # port_marker = filtered_events[ind-2, 0] #needs to change
        # print(port_marker)
        # port = port_lookup[port_marker]
        datetime = filtered_events[ind, 5]
        task = filtered_events[ind, 2]
        event = filtered_events[ind, 0]

        # select events occurring after trialStartMarker
        nextEvents = filtered_events[ind:]
        # print(nextEvents[0:100, :])
        try:
            # find the index of trialMissMarker or trialHitMarker in nextEvents
            status_indices_ports = np.where(
                (nextEvents[:, 0] == trialStartMarker) & (nextEvents[:, 1] == 1)
            )[0]
            # print(status_indices_ports)
            status_index_ports = (
                status_indices_ports[0] if status_indices_ports.size > 0 else None
            )
            status_indices = np.where(
                (nextEvents[:, 0] == trialMissMarker)
                | (nextEvents[:, 0] == trialHitMarker)
            )[0]
            statusIndex = status_indices[0] if status_indices.size > 0 else None

            # find the index of trialProbMarker in nextEvents

            # contLine: if np.any(nextEvents[:, 0] == trialProbMarker) else None

            if statusIndex is not None:
                # find reward, trialend, and rewprob based on identified events
                # print(statusIndex)
                # print(nextEvents[statusIndex,0])
                reward = event_lookup[nextEvents[statusIndex, 0]]
                trialend = nextEvents[statusIndex, 4]
                # rewprob = nextEvents[rewprob_index, 1] if rewprob_index is not None else np.nan
            else:
                # if trialMissMarker or trialHitMarker not found, nan
                reward = np.nan
                trialend = np.nan
                # rewprob = np.nan
            if status_index_ports is not None:
                # find reward, trialend, and rewprob based on identified events
                # print(status_index_ports)
                # print(status_index_ports-2)
                # print(nextEvents[(status_index_ports-1), 0])
                port = port_lookup[nextEvents[(status_index_ports - 1), 0]]
            else:
                port = np.nan
        except IndexError:
            # set values to nan in case of issues
            reward = np.nan
            trialend = np.nan
            port = np.nan
            # rewprob = np.nan

        # append to sess_list
        sess_list.append(
            [
                npcounter,
                trialstart,
                port,
                reward,
                trialend,
                sessnum,
                datetime,
                task,
                event,
                rewprobl,
                rewprobl2,
            ]
        )
        npcounter += 1

        if filtered_events[ind, 0] == sessionStartMarker:
            # update session number when sessionStartMarker found
            rewprobl = 0
            rewprobl2 = 0
            sessnum += 1

        if (filtered_events[ind, 0] == rewardProbMarker1) & (
            filtered_events[ind, 1] == 1
        ):
            rewprobl = filtered_events[ind, 4]

        if (filtered_events[ind, 0] == rewardProbMarker2) & (
            filtered_events[ind, 1] == 2
        ):
            rewprobl2 = filtered_events[ind, 4]

    # create pandas dataframe from the list of all session data
    sessdf = pd.DataFrame(
        sess_list,
        columns=[
            "trial#",
            "trialstart",
            "port",
            "reward",
            "trialend",
            "session#",
            "eptime",
            "task",
            "event",
            "rewprobfull1",
            "rewprobfull2",
        ],
    )

    # remove all lines corresponding to start of a session or reward prob marker
    sessdf = (
        sessdf[
            (sessdf.event != sessionStartMarker)
            & (sessdf.event != rewardProbMarker1)
            & (sessdf.event != rewardProbMarker2)
        ]
        .reset_index(drop=True)
        .drop(columns="event")
    )

    # remove all lines that have (trialend-trialstart)> 5500 or negative (sum of tone + wait time)
    # is this check required?

    # reset trial numbers
    sessdf = sessdf.dropna()
    sessdf.reset_index()
    sessdf["trial#"] = np.arange(len(sessdf))

    # add animal information
    sessdf["animal"] = animal

    # add datetime
    sessdf["datetime"] = pd.to_datetime(sessdf.eptime, unit="s")

    return sessdf


def trializer_v3_test_reversePorts(
    df,
    list_sessionMarker,
    trialStartMarker,
    trialHitMarker,
    trialMissMarker,
    sessionStartMarker,
    rewardProbMarker1,
    rewardProbMarker2,
    nosePokeMarker1,
    nosePokeMarker2,
    animal,
    arms=2,
):
    """Function to convert a df of joined .dat files with column names in order :
    'event', 'value', 'sm', 'tp', 'smtime', 'eptime', 'dw', 'ww'

    INPUTS: df, list of session state machine numbers, trial start marker, trial rewarded marker,
    trial miss/unrewarded marker, session start marker, trial probability marker, session reward prob marker,
    animal name, num arms in task

    OUTPUT: single trial-wise dataframe of one animal
    2023-10-26
    """
    # initialize var
    sessnum = 0
    npcounter = 0
    sess_list = []
    rewprobl = 0
    rewprobl2 = 0
    event_lookup = {
        trialHitMarker: 1,
        trialMissMarker: 0,
    }
    port_lookup = {
        nosePokeMarker1: 2,
        nosePokeMarker2: 1,
        trialMissMarker: np.nan,
        trialHitMarker: np.nan,
        83: np.nan,
        trialStartMarker: np.nan,
    }
    list_events = [
        trialStartMarker,
        trialHitMarker,
        trialMissMarker,
        sessionStartMarker,
        rewardProbMarker1,
        rewardProbMarker2,
        nosePokeMarker1,
        nosePokeMarker2,
    ]

    # filter events as np array to increase speed and avoid repetitive indexing
    filtered_events = df[
        (df["sm"].isin(list_sessionMarker)) & (df["event"].isin(list_events))
    ].to_numpy()
    # print(filtered_events[0:50, :])
    indices = np.where(
        (filtered_events[:, 0] == trialStartMarker)
        | (filtered_events[:, 0] == sessionStartMarker)
        | (filtered_events[:, 0] == rewardProbMarker1)
        | (filtered_events[:, 0] == rewardProbMarker2)
    )[
        0
    ]  # indices of events

    for ind in indices:
        # extract information from the current trialStartMarker
        trialstart = filtered_events[ind, 4]
        # port_marker = filtered_events[ind-2, 0] #needs to change
        # print(port_marker)
        # port = port_lookup[port_marker]
        datetime = filtered_events[ind, 5]
        task = filtered_events[ind, 2]
        event = filtered_events[ind, 0]

        # select events occurring after trialStartMarker
        nextEvents = filtered_events[ind:]
        # print(nextEvents[0:100, :])
        try:
            # find the index of trialMissMarker or trialHitMarker in nextEvents
            status_indices_ports = np.where(
                (nextEvents[:, 0] == trialStartMarker) & (nextEvents[:, 1] == 1)
            )[0]
            # print(status_indices_ports)
            status_index_ports = (
                status_indices_ports[0] if status_indices_ports.size > 0 else None
            )
            status_indices = np.where(
                (nextEvents[:, 0] == trialMissMarker)
                | (nextEvents[:, 0] == trialHitMarker)
            )[0]
            statusIndex = status_indices[0] if status_indices.size > 0 else None

            # find the index of trialProbMarker in nextEvents

            # contLine: if np.any(nextEvents[:, 0] == trialProbMarker) else None

            if statusIndex is not None:
                # find reward, trialend, and rewprob based on identified events
                # print(statusIndex)
                # print(nextEvents[statusIndex,0])
                reward = event_lookup[nextEvents[statusIndex, 0]]
                trialend = nextEvents[statusIndex, 4]
                # rewprob = nextEvents[rewprob_index, 1] if rewprob_index is not None else np.nan
            else:
                # if trialMissMarker or trialHitMarker not found, nan
                reward = np.nan
                trialend = np.nan
                # rewprob = np.nan
            if status_index_ports is not None:
                # find reward, trialend, and rewprob based on identified events
                # print(status_index_ports)
                # print(status_index_ports-2)
                # print(nextEvents[(status_index_ports-1), 0])
                port = port_lookup[nextEvents[(status_index_ports - 1), 0]]
            else:
                port = np.nan
        except IndexError:
            # set values to nan in case of issues
            reward = np.nan
            trialend = np.nan
            port = np.nan
            # rewprob = np.nan

        # append to sess_list
        sess_list.append(
            [
                npcounter,
                trialstart,
                port,
                reward,
                trialend,
                sessnum,
                datetime,
                task,
                event,
                rewprobl,
                rewprobl2,
            ]
        )
        npcounter += 1

        if filtered_events[ind, 0] == sessionStartMarker:
            # update session number when sessionStartMarker found
            rewprobl = 0
            rewprobl2 = 0
            sessnum += 1

        if (filtered_events[ind, 0] == rewardProbMarker1) & (
            filtered_events[ind, 1] == 1
        ):
            rewprobl = filtered_events[ind, 4]

        if (filtered_events[ind, 0] == rewardProbMarker2) & (
            filtered_events[ind, 1] == 2
        ):
            rewprobl2 = filtered_events[ind, 4]

    # create pandas dataframe from the list of all session data
    sessdf = pd.DataFrame(
        sess_list,
        columns=[
            "trial#",
            "trialstart",
            "port",
            "reward",
            "trialend",
            "session#",
            "eptime",
            "task",
            "event",
            "rewprobfull1",
            "rewprobfull2",
        ],
    )

    # remove all lines corresponding to start of a session or reward prob marker
    sessdf = (
        sessdf[
            (sessdf.event != sessionStartMarker)
            & (sessdf.event != rewardProbMarker1)
            & (sessdf.event != rewardProbMarker2)
        ]
        .reset_index(drop=True)
        .drop(columns="event")
    )

    # remove all lines that have (trialend-trialstart)> 5500 or negative (sum of tone + wait time)
    # is this check required?

    # reset trial numbers
    sessdf = sessdf.dropna()
    sessdf.reset_index()
    sessdf["trial#"] = np.arange(len(sessdf))

    # add animal information
    sessdf["animal"] = animal

    # add datetime
    sessdf["datetime"] = pd.to_datetime(sessdf.eptime, unit="s")

    return sessdf


def rew_prob_extractor(df, arms, ProbMarker1, ProbMarker2):
    """Extracts reward prob as dict from given df for set number of bandit arms.
    Input: df of .dat file, number of arms, rewardProbMarker
    Output: rewardProb dict
    """
    # extract all rew prob as dict
    rewardProb = {}
    session_count = 0
    regret = []
    temp = []

    for ind, prob in enumerate(df[df[0] == rewardProbMarker][4]):
        temp.append(prob)
        rewardProb[session_count] = temp

        if (ind + 1) % arms == 0:
            session_count += 1
            temp = []
    return rewardProb


def rew_prob_extractor_v2(df, arms, probMarker1, probMarker2):
    Prob1 = {}
    Prob2 = {}
    session_count = 0
    temp = 0
    regret = []

    for ind, prob in enumerate(
        df[(df["event"] == probMarker1) & (df["value"] == 1)]["smtime"]
    ):
        temp = prob
        Prob1[session_count] = temp
        session_count += 1

    for ind, prob in enumerate(
        df[(df["event"] == probMarker2) & (df["value"] == 2)]["smtime"]
    ):
        temp = prob
        Prob2[session_count] = temp
        session_count += 1

    return Prob1, Prob2


def add_uneq_vec(a, b):
    if len(a) < len(b):
        c = b.copy()
        c = np.sum((c[: len(a)], a), axis=0)
    else:
        c = a.copy()
        c = np.sum((c[: len(b)], b), axis=0)
    return c


def add_reward_vectors(sessdf):
    sessions = pd.unique(sessdf["session#"])  # list of all sessions
    # print(sessions)
    trials = []
    index = 2
    for e in sessions:
        t = sessdf.loc[lambda sessdf: sessdf["session#"] == e, ["reward"]]
        tl = t["reward"].tolist()
        # print(tl)
        trials.append(tl)

    sumv = add_uneq_vec(trials[0], trials[1])
    # print(sumv)
    sumv = add_uneq_vec(sumv, trials[2])
    # print(sumv)
    for i in range(len(sessions) - 3):
        sumv = add_uneq_vec(sumv, trials[index + 1])
        # print(sumv)
        index += 1
    return sumv / len(sessions)


def regret(df, min_length, xlim):
    df = df.groupby("session#").filter(lambda x: len(x["trial#"]) > min_length)
    # print(len(pd.unique(df['session#'])))
    df_mean = df.groupby("session#").mean(numeric_only=True)
    # print(df_mean)
    best = []
    count = 2
    for i, row in df_mean.iterrows():
        r1 = df_mean["rewprobfull1"][i]
        r2 = df_mean["rewprobfull2"][i]
        best.append(np.maximum(r1, r2))
    # print(best)
    sessions = pd.unique(df["session#"])
    regret_aray = []
    for index, e in enumerate(sessions):
        choice = df.loc[lambda df: df["session#"] == e, ["rw"]].to_numpy().flatten()
        regret = abs(np.subtract(np.ones(len(choice)) * best[index], choice)) * 0.01
        # print(regret)
        regret_aray.append(regret[:xlim])
    # print(regret_aray)

    sumr = add_uneq_vec(regret_aray[0], regret_aray[1])
    # print('sumr1 ', sumr)
    # print('ra2', regret_aray[2])
    sumr = add_uneq_vec(sumr, regret_aray[2])
    # print('sumr2 ', sumr)
    for i in range(len(sessions) - 3):
        # print(count)
        # print(sumr)
        # print(regret_aray[count+1])
        sumr = add_uneq_vec(sumr, regret_aray[count + 1])

        # print('sum', sumr)
        count += 1
    return sumr / len(sessions)


def hard_regret(df, min_length, xlim):
    df = df.groupby("session#").filter(lambda x: len(x["trial#"]) > min_length)
    print(len(pd.unique(df["session#"])))
    df_mean = df.groupby("session#").mean(numeric_only=True)
    # print(df_mean)
    best = []
    count = 2
    for i, row in df_mean.iterrows():
        r1 = df_mean["rewprobfull1"][i]
        r2 = df_mean["rewprobfull2"][i]
        best.append(np.maximum(r1, r2))
    # print(best)
    sessions = pd.unique(df["session#"])
    regret_aray = []
    for index, e in enumerate(sessions):
        choice = df.loc[lambda df: df["session#"] == e, ["rw"]].to_numpy().flatten()
        regret = abs(np.subtract(np.ones(len(choice)) * best[index], choice)) * 0.01
        for i in range(len(regret)):
            if regret[i] > 0:
                regret[i] = 1
        # print(regret)
        regret_aray.append(regret[:xlim])

    sumr = add_uneq_vec(regret_aray[0], regret_aray[1])
    # print('sumr1 ', sumr)
    # print('ra2', regret_aray[2])
    # print(regret_aray[2])
    sumr = add_uneq_vec(sumr, regret_aray[2])
    # print('sumr2 ', sumr)
    for i in range(len(sessions) - 3):
        # print(count)
        # print(sumr)
        # print(regret_aray[count+1])
        sumr = add_uneq_vec(sumr, regret_aray[count + 1])

        # print('sum', sumr)
        count += 1
    return sumr / len(sessions)  # return averaged regret


def sessdf_unstr(
    animal, filepath, rewProbMarker2
):  ## Output sessdf for unstr task #rewProbMarker 2 = 83
    files = get_files(log_folder=filepath, extension=".dat")
    df = merge_files_to_df(files)
    sessdf = trializer_v3_test_reversePorts(
        df, [13], 23, 51, 86, 61, 83, rewProbMarker2, 20, 21, animal, arms=2
    )
    sessdf = sessdf.reset_index()
    sessdf = sessdf.drop("index", axis=1)
    list = []
    for n in range(len(sessdf["trial#"])):
        if sessdf["port"][n] == 1.0:
            list.append(sessdf["rewprobfull1"][n])
        elif sessdf["port"][n] == 2.0:
            list.append(sessdf["rewprobfull2"][n])
    sessdf.insert(10, "rw", list)
    return sessdf


def sessdf_str(
    animal, filepath, rewProbMarker2
):  ## Output sessdf for str task #rewProbMarker2 = 84
    files = get_files(log_folder=filepath, extension=".dat")
    df = merge_files_to_df(files)
    sessdf = trializer_v3_test_reversePorts(
        df, [12], 23, 51, 86, 61, 83, rewProbMarker2, 20, 21, animal, arms=2
    )
    sessdf = sessdf.reset_index()
    sessdf = sessdf.drop("index", axis=1)
    list = []
    for n in range(len(sessdf["trial#"])):
        if sessdf["port"][n] == 1.0:
            list.append(sessdf["rewprobfull1"][n])
        elif sessdf["port"][n] == 2.0:
            list.append(sessdf["rewprobfull2"][n])
    sessdf.insert(10, "rw", list)
    return sessdf


def rr_grid(
    sessdf, n_rows, max_rew, y_min=0.5, y_max=0.85
):  # series of reward-rate plots with increasing session lengths
    l = 50  ##
    fig, axs = plt.subplots(n_rows, 2, figsize=(15, 15))

    for row in range(n_rows):
        for column in range(2):
            data1 = sessdf.groupby("session#").filter(lambda x: len(x["trial#"]) > l)
            # print(len(data1.groupby('session#')))
            mov_avg1 = (
                pd.DataFrame(add_reward_vectors(data1)).rolling(30, center=True).mean()
            )
            axs[row][column].plot(
                range(len(mov_avg1[0].to_list())), mov_avg1[0].to_list(), "b"
            )
            axs[row][column].plot(
                range(len(mov_avg1[0].to_list())),
                np.ones(len(mov_avg1[0].to_list())) * max_rew,
                "k",
            )
            axs[row][column].set_xlim(0, l)
            axs[row][column].set_ylim(y_min, y_max)
            axs[row][column].set_xlabel("trial#")
            axs[row][column].set_ylabel("reward rate")
            column += 1
            l += 50
            # print (l)
        row += 1
    sns.despine()
    plt.show()


def bias_plot(
    sessdf, min_length=100, tail=50
):  ## doesn't work || Dividing by zero error
    # fraction of choices where port 1 was selected, given contrasts ranging from -80 to +80
    sessdf_c = (
        sessdf.groupby("session#")
        .filter(lambda x: len(x["trial#"]) > min_length)
        .groupby("session#")
        .tail(tail)
    )
    sessdf_c.groupby("session#").mean(numeric_only=True)
    c1, c2 = (
        sessdf_c[(sessdf_c["rewprobfull1"] - sessdf_c["rewprobfull2"]) == -80],
        sessdf_c[(sessdf_c["rewprobfull1"] - sessdf_c["rewprobfull2"]) == -60],
    )
    c3, c4 = (
        sessdf_c[(sessdf_c["rewprobfull1"] - sessdf_c["rewprobfull2"]) == -40],
        sessdf_c[(sessdf_c["rewprobfull1"] - sessdf_c["rewprobfull2"]) == -20],
    )
    c5, c6 = (
        sessdf_c[(sessdf_c["rewprobfull1"] - sessdf_c["rewprobfull2"]) == 20],
        sessdf_c[(sessdf_c["rewprobfull1"] - sessdf_c["rewprobfull2"]) == 40],
    )
    c7, c8 = (
        sessdf_c[(sessdf_c["rewprobfull1"] - sessdf_c["rewprobfull2"]) == 60],
        sessdf_c[(sessdf_c["rewprobfull1"] - sessdf_c["rewprobfull2"]) == 80],
    )
    array = [c1, c2, c3, c4, c5, c6, c7, c8]

    for i, df in enumerate(array):  # prevent zero division error
        if (len(df) == 0) | (
            len(pd.unique(df["session#"])) < 3
        ):  # dont consider df is number of sessions is less than 3
            array[i] = pd.DataFrame(
                {"port": np.zeros(10000)}
            )  # dividing by a large number to return zero

    C1, C2, C3, C4, C5, C6, C7, C8 = (
        len(c1[c1["port"] == 1]) / len(array[0]),
        len(c2[c2["port"] == 1]) / len(array[1]),
        len(c3[c3["port"] == 1]) / len(array[2]),
        len(c4[c4["port"] == 1]) / len(array[3]),
        len(c5[c5["port"] == 1]) / len(array[4]),
        len(c6[c6["port"] == 1]) / len(array[5]),
        len(c7[c7["port"] == 1]) / len(array[6]),
        len(c8[c8["port"] == 1]) / len(array[7]),
    )  # ignore any values that are undefined due to zero division?
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        [-80, -60, -40, -20, 20, 40, 60, 80],
        np.multiply([C1, C2, C3, C4, C5, C6, C7, C8], np.ones(8) * 100),
        "b",
    )
    ax.plot(np.ones(100) * 0, range(100), "k", linewidth=0.4)
    ax.plot([-80, -60, -40, -20, 20, 40, 60, 80], np.ones(8) * 50, "k", linewidth=0.4)
    plt.xlabel("contrast (port1-port2)")
    plt.ylabel("% choose port 1")
    plt.xticks((-80, -60, -40, -20, 20, 40, 60, 80))
    plt.yticks(np.arange(0, 105, step=5))
    plt.ylim(0, 105)
    sns.despine()

    return np.multiply([C1, C2, C3, C4, C5, C6, C7, C8], np.ones(8) * 100)


def choice_unstr(
    sessdf, trial_cutoff, n_plots
):  ## trial-by-trial choice plot for unstructured task
    """
    return session-wise choice plot for unstr data.
    sessdf - input dataframe
    trial_cutoff - min session length allowed
    n_plots - number of plots to create (depends on the number of sessions available)
    """
    # use reward prob as hue
    # in sessdf, add reward prob, with each entry corresponding to the port selected
    sessdf = sessdf.groupby("session#").filter(
        lambda x: len(x["trial#"]) > trial_cutoff
    )

    colors = [
        "",
        "#f1ee8e",
        "#f5c77e",
        "#f69697",
        "#c30010",
        "#d24e01",
        "#e69b00",
        "#008631",
    ]
    # green, yellow, orange, red (light and dark)
    palette = sns.cubehelix_palette(start=0, rot=-0.75, dark=0, light=0.8, as_cmap=True)
    x = 0
    y = 20
    hue = sessdf["rw"].sort_values()

    for i in range(n_plots):
        session_subset = sessdf[sessdf["session#"].isin(range(x, y))]
        # print(session_subset)
        # session_subset['rw'] = session_subset['rw'].map(lambda x: str(x))
        session_subset
        # palette = sns.color_palette("hls", 8)
        markers = {0: "$o$", 1: "o"}
        # hue_order = [10, 20, 30, 40, 60, 70, 80, 90]
        g = sns.relplot(
            data=session_subset,
            x="trial#",
            y="port",
            aspect=5,
            linewidth=0.1,
            style="reward",
            markers=markers,
            s=150,
            legend="full",
            palette="rocket_r",
            hue=hue,
        )

        last = session_subset.groupby("session#")["trial#"].max()
        axes = g.axes.flatten()
        for ax in axes:
            for ind, l in enumerate(last):
                ax.axvline(l, linewidth=2, color="grey")
                ax.set_yticks([1, 2])

        plt.ylim(0.5, 2.5)
        plt.title("", fontsize=20)
        g._legend.texts[0].set_text("Reward %")
        # g._legend.texts[5].set_text("Outcome")
        plt.xlabel("Trials")
        plt.ylabel("Port")
        x += 20
        y += 20


def combined_regret(
    sessdfList,
    animalList,
    exclude_sessions,
    min_length,
    regret_window,
    xlim,
    draw_subset,
    hard_soft,
    title,
):
    """Calculates average regret for all animals and plots it on the same figure
    animalList: list of animals in order: structured, unstructured
    sessdfList: list of sessions-wise dfs in order: structured, unstructured
    exclued_sessions: exclude first n sessions
    min_length: min length of a session to be included in regret calculation
    regret_window: pd.rolling() moving window
    """
    regret_list = []
    for e in sessdfList:
        # extracting sessions with certain probabilities:
        e_subset = (
            e[(e["session#"] > exclude_sessions)]
            .groupby("session#")
            .mean(numeric_only=True)
        )
        p1 = e_subset["rewprobfull1"]
        p2 = e_subset["rewprobfull2"]
        allowed_probs = set([10, 90])
        allowed_probs2 = set([20, 80])
        allowed_probs3 = set([30, 70])
        allowed_probs4 = set([40, 60])

        keep = []
        for i, row in e_subset.iterrows():
            if (
                (allowed_probs == set([p1[i], p2[i]]))
                | (allowed_probs2 == set([p1[i], p2[i]]))
                | (allowed_probs3 == set([p1[i], p2[i]]))
                | (allowed_probs4 == set([p1[i], p2[i]]))
            ):
                keep.append(i)
        # print('len of keep ', len(keep))
        # print(keep)
        if draw_subset == True:
            df_subset = e[e["session#"].isin(keep)]
        else:
            df_subset = e[(e["session#"] > exclude_sessions)]

        data_subset = df_subset.groupby("session#").filter(
            lambda x: len(x["trial#"]) > min_length
        )
        # print(data_subset)
        if hard_soft == "soft":
            mov_avg_subset_regret = (
                pd.DataFrame(regret(data_subset, min_length, xlim))
                .rolling(regret_window, center=True)
                .mean()
            )
        elif hard_soft == "hard":
            mov_avg_subset_regret = (
                pd.DataFrame(hard_regret(data_subset, min_length, xlim))
                .rolling(regret_window, center=True)
                .mean()
            )
        regret_list.append(mov_avg_subset_regret)

    fig, axs = plt.subplots()
    axs.plot(range(len(regret_list[0])), regret_list[0], "#5580ff")
    axs.plot(range(len(regret_list[1])), regret_list[1], "b")
    axs.plot(range(len(regret_list[2])), regret_list[2], "#ff2055")
    axs.plot(range(len(regret_list[3])), regret_list[3], "r")
    if hard_soft == "hard":
        axs.set_ylabel("hard regret")
    elif hard_soft == "soft":
        axs.set_ylabel("soft regret")

    axs.set_xlabel("trial#")

    plt.title(title)
    plt.xlim(0, xlim)
    plt.legend((animalList), loc="best")
    sns.despine()


def choice_str(
    sessdf, trial_cutoff, n_plots
):  ## trial-by-trial chocie plot for structured task
    # use reward prob as hue
    # in sessdf, add reward prob, with each entry corresponding to the port selected
    sessdf = sessdf.groupby("session#").filter(
        lambda x: len(x["trial#"]) > trial_cutoff
    )

    colors = [
        "#cce7c9",
        "#f1ee8e",
        "#f5c77e",
        "#f69697",
        "#c30010",
        "#d24e01",
        "#e69b00",
        "#008631",
    ]
    # green, yellow, orange, red (light and dark)
    sns.set_palette(sns.color_palette(colors))
    x = 0
    y = 20
    hue = sessdf["rw"].sort_values()

    for i in range(n_plots):
        session_subset = sessdf[sessdf["session#"].isin(range(x, y))]
        # print(session_subset)
        # session_subset['rw'] = session_subset['rw'].map(lambda x: str(x))
        session_subset
        # palette = sns.color_palette("hls", 8)
        markers = {0: "$o$", 1: "o"}
        # hue_order = [10, 20, 30, 40, 60, 70, 80, 90]
        g = sns.relplot(
            data=session_subset,
            x="trial#",
            y="port",
            aspect=5,
            linewidth=0.1,
            hue=hue,
            style="reward",
            markers=markers,
            s=150,
            legend="full",
            palette=sns.color_palette(colors, 8),
        )
        last = session_subset.groupby("session#")["trial#"].max()
        axes = g.axes.flatten()
        for ax in axes:
            for ind, l in enumerate(last):
                ax.axvline(l, linewidth=2, color="grey")
                ax.set_yticks([1, 2])

        plt.ylim(0.5, 2.5)
        plt.title("", fontsize=20)
        g._legend.texts[0].set_text("Reward %")
        # g._legend.texts[5].set_text("Outcome")
        plt.xlabel("Trials")
        plt.ylabel("Port")
        x += 30
        y += 30


def filter_by_time(df, t_diff=10800):
    """
    input: session-wise animal df
    t_diff: minimum permitted time diff between sessions in seconds (default = 10800 / 3 hours)
    """
    df_mod = df.copy()
    time = []
    for i in range(len(df_mod.datetime)):
        t = pd.to_datetime(df_mod.datetime[i]).time()
        seconds = (t.hour * 60 + t.minute) * 60 + t.second
        time.append(seconds)
    df_mod.insert(14, "seconds", time)
    keep_sess = []
    heads = df_mod.groupby("session#").head(1).reset_index(drop=True)
    for i in range(len(heads) - 1):
        if abs(heads["seconds"][i + 1] - heads["seconds"][i]) >= t_diff:
            keep_sess.append(i + 1)
    df_mod = df_mod[df_mod["session#"].isin(keep_sess)]
    return df_mod


def subset(df):
    return df[
        ((df["rewprobfull1"] == 10) & (df["rewprobfull2"] == 90))
        | ((df["rewprobfull1"] == 20) & (df["rewprobfull2"] == 80))
        | ((df["rewprobfull1"] == 30) & (df["rewprobfull2"] == 70))
        | ((df["rewprobfull1"] == 40) & (df["rewprobfull2"] == 60))
        | ((df["rewprobfull1"] == 90) & (df["rewprobfull2"] == 10))
        | ((df["rewprobfull1"] == 80) & (df["rewprobfull2"] == 20))
        | ((df["rewprobfull1"] == 70) & (df["rewprobfull2"] == 30))
        | ((df["rewprobfull1"] == 60) & (df["rewprobfull2"] == 40))
    ]


def subset_unstructured(df):
    """remove all rewprobs possible in the structured environment"""
    return df[
        ((df["rewprobfull1"] != 10) & (df["rewprobfull2"] != 90))
        | ((df["rewprobfull1"] != 20) & (df["rewprobfull2"] != 80))
        | ((df["rewprobfull1"] != 30) & (df["rewprobfull2"] != 70))
        | ((df["rewprobfull1"] != 40) & (df["rewprobfull2"] != 60))
        | ((df["rewprobfull1"] != 90) & (df["rewprobfull2"] != 10))
        | ((df["rewprobfull1"] != 80) & (df["rewprobfull2"] != 20))
        | ((df["rewprobfull1"] != 70) & (df["rewprobfull2"] != 30))
        | ((df["rewprobfull1"] != 60) & (df["rewprobfull2"] != 40))
    ]


def subset_difference(df, condition):
    if condition == "easy":
        return df[abs(df["rewprobfull1"] - df["rewprobfull2"]) >= 50]
    if condition == "hard":
        return df[abs(df["rewprobfull1"] - df["rewprobfull2"]) <= 40]


# learning over time:
# plot (cumm?)regret reached at the end of n trials
def longitudinal(
    df,
    order,
    window=5,
    head=150,
    tail=20,
    title="",
    subset=False,
    condition="easy",
    hline=0,
):
    """longitudinal analysis representing learning over time. plots mean regret achieved during the last n trials against session#
    INPUTS:
    df - unmodified animal dataframe
    order - 'g'/'r' : structured first:ustructured first
    window - for pd.rolling()
    head - take first n trials of each session
    tail - take last n trials of the subset obtained after applying head()
    title - plot title
    subset - If True, subset sessions based on difficulty
    condition - applied to subset (possible values: 'easy'/'hard')
    OUTPUT:
    regret vs session# plot
    """

    if subset == True:
        if condition == "easy":
            df = subset_difference(df, "easy")
        if condition == "hard":
            df = subset_difference(df, "hard")
    df_tails = df.groupby("session#").head(head).groupby("session#").tail(tail)
    sessions = pd.unique(df_tails["session#"])
    df_mean = df_tails.groupby("session#").mean(numeric_only=True)
    best = []
    for i, row in df_mean.iterrows():
        r1 = df_mean["rewprobfull1"][i]
        r2 = df_mean["rewprobfull2"][i]
        best.append(np.maximum(r1, r2))
    regret_list = []
    for index, e in enumerate(sessions):
        choice = (
            df_tails.loc[lambda df_tails: df_tails["session#"] == e, ["rw"]]
            .to_numpy()
            .flatten()
        )
        regret = abs(np.subtract(np.ones(len(choice)) * best[index], choice)) * 0.01
        for i in range(len(regret)):
            if regret[i] > 0:
                regret[i] = 1
        regret = np.mean(regret)
        regret_list.append(regret)
    regret_list
    fig, ax = plt.subplots(layout="tight")
    ax.plot(
        range(len(regret_list)),
        pd.DataFrame(regret_list).rolling(window, center=True).mean(),
        "b",
    )
    if hline > 0:
        # ax.plot(np.ones(8)*hline, np.arange(0, 0.8, 0.1), 'k')
        if order == "g":
            ax.axvspan(xmin=0, xmax=hline, color="g", alpha=0.2)
            ax.axvspan(
                xmin=hline,
                xmax=np.max(df.groupby("session#").mean(numeric_only=True).index),
                color="r",
                alpha=0.2,
            )
        if order == "r":
            ax.axvspan(xmin=0, xmax=hline, color="r", alpha=0.2)
            ax.axvspan(
                xmin=hline,
                xmax=np.max(df.groupby("session#").mean(numeric_only=True).index),
                color="g",
                alpha=0.2,
            )

    plt.title(title)
    plt.xlabel("Session #")
    plt.ylabel("regret")
    plt.xlim(0, len(pd.unique(df["session#"])))
    plt.ylim(0, 0.7)
    sns.despine()
    plt.show()


def add_cols(df, rat):
    """
    modify df to make it compatible with celiaberon logreg code
    add columns: blockTrial, blockLength, Target
    modify column names
    input: df, rat(str, eg. R1)
    output: modified df
    """
    df = df.rename(
        columns={
            "trial#": "Trial",
            "port": "Decision",
            "reward": "Reward",
            "session#": "Session",
        }
    )
    df["Decision"] = df["Decision"].replace({1: 0, 2: 1})
    switches = abs(
        df["Decision"].shift(
            -1,
        )
        - df["Decision"]
    )
    switches.iloc[-1] = 0
    df.insert(4, "Switch", switches)
    #### insert blockTrials:
    sessions = pd.unique(df["Session"])
    df.insert(2, "blockTrial", np.zeros(len(df)))

    for i, session in enumerate(sessions):
        if i > 0:
            blockTrial = df[df["Session"] == session]["Trial"].sub(
                (
                    np.ones(len(df[df["Session"] == session]))
                    * max(df[df["Session"] == sessions[i - 1]]["Trial"])
                )
            )
        elif i == 0:
            blockTrial = df[df["Session"] == session]["Trial"]
        df.loc[df["Session"] == session, ["blockTrial"]] = blockTrial

    #### insert blockLength
    df.insert(3, "blockLength", np.zeros(len(df)))

    for session in sessions:
        blockLength = np.ones(len(df[df["Session"] == session])) * max(
            df[df["Session"] == session]["blockTrial"]
        )
        df.loc[df["Session"] == session, ["blockLength"]] = blockLength

    #### insert Target
    df.insert(4, "Target", np.zeros(len(df)))
    for session in sessions:
        if np.mean(df[df["Session"] == session]["rewprobfull1"]) > np.mean(
            df[df["Session"] == session]["rewprobfull2"]
        ):
            target = 1
        else:
            target = 2
        df.loc[df["Session"] == session, ["Target"]] = target
    df["Session"] = df.Session.map(lambda x: rat + "_" + f"{x}")
    df["Rat"] = rat
    return df


##target
