import os
from copy import deepcopy
from pprint import pprint
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from obspy.core.stream import Stream
from scipy import signal

from src.constants import FONT_SIZE, WAVEFORM_YLIM
from src.utils import get_datetime_ticks

plt.rcParams["font.size"] = FONT_SIZE


def calc_sta_lta(
    stream: Stream,
    tl_sec: float = 1800,  # window size of LTA (hyperparameter)
    ts_sec: float = 300,  # window size of STA (hyperparameter)
) -> ndarray:
    sampling_rate = stream[0].stats.sampling_rate
    d = pd.Series(stream[0].data).abs()

    tl, ts = round(sampling_rate * tl_sec), round(sampling_rate * ts_sec)

    sta, lta = d.rolling(ts).mean(), d.rolling(tl).mean()
    sta, lta = sta[~pd.isnull(sta)], lta[~pd.isnull(lta)]

    sta = sta.iloc[tl:]
    lta = lta.iloc[: sta.shape[0]]

    sta, lta = np.array(sta), np.array(lta)
    result = sta / lta
    result = result * signal.cosine(len(result))

    return result


def get_args_over_sta_lta_threshold(
    stream: Stream,
    input_sta_lta: Optional[ndarray] = None,
    sta_lta_threshold: float = 10,
    run_length_threshold: Optional[float] = None,
    plot_title: Optional[str] = None,
    plot_filename: Optional[str] = None,
) -> tuple[ndarray, ndarray]:
    if input_sta_lta is None:
        sta_lta = calc_sta_lta(stream=stream)
    else:
        sta_lta = input_sta_lta.copy()

    comp_sec, lengths = _calc_run_length(sta_lta >= sta_lta_threshold)
    run_length = lengths[comp_sec]

    start_args, end_args = (
        np.array(
            [
                np.where(sta_lta >= sta_lta_threshold)[0][np.sum(run_length[:i])]
                for i in range(len(run_length))
            ]
        ),
        np.array(
            [
                np.where(sta_lta >= sta_lta_threshold)[0][
                    np.sum(run_length[: i + 1]) - 1
                ]
                for i in range(len(run_length))
            ]
        ),
    )

    if run_length_threshold is not None:
        target_args = run_length >= run_length_threshold
        start_args = start_args[target_args]
        end_args = end_args[target_args]

    if plot_title or plot_filename:
        fig = plt.figure(figsize=(24, 8))
        ax1 = fig.add_subplot(1, 1, 1)

        # plot STA/LTA
        sta_lta_over_threshold = deepcopy(sta_lta)
        sta_lta_over_threshold[sta_lta < sta_lta_threshold] = None

        ax1.set_ylabel("STA/LTA", color="blue")
        ax1.plot(sta_lta, color="blue")
        ax1.plot(sta_lta_over_threshold, color="red")
        ax1.plot(np.repeat(sta_lta_threshold, len(sta_lta)), color="#32CD32")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.set_ylim(-1, 13)
        ax1.grid(which="major", axis="y", linestyle="dotted")

        # plot signal
        ax2 = ax1.twinx()
        ax2.set_ylabel("|m/s|", color="black")
        ax2.plot(
            pd.Series(stream[0].data[-len(sta_lta) :]).abs(), color="black", alpha=0.2
        )
        ax2.set_ylim(0, WAVEFORM_YLIM)

        # visualize start line
        ax1.vlines(start_args, 0, np.max(sta_lta), color="cyan")

        fig.tight_layout()
        ticks, datetime_ticks = get_datetime_ticks(stream)
        plt.xticks(ticks, datetime_ticks)
        if plot_title:
            plt.title(plot_title)
        if plot_filename:
            os.makedirs("/".join(plot_filename.split("/")[:-1]), exist_ok=True)
            plt.savefig(plot_filename)
        else:
            plt.show()

    return start_args, end_args


def get_profile_details(
    stream: Stream,
    input_sta_lta: Optional[ndarray] = None,
    start_args: ndarray = np.array([]),
    end_args: ndarray = np.array([]),
    plot_title: Optional[str] = None,
    plot_filename: Optional[str] = None,
) -> list[dict]:
    if len(start_args) != len(end_args):
        raise Exception("starts != ends")

    if input_sta_lta is None:
        sta_lta = calc_sta_lta(stream=stream)
    else:
        sta_lta = input_sta_lta.copy()

    profile_details = []
    for start_arg, end_arg in zip(start_args, end_args):
        profile_argmax = start_arg + np.argmax(sta_lta[start_arg : end_arg + 1])
        profile_max = np.max(sta_lta[start_arg : end_arg + 1])
        profile_half = profile_max / 2

        # calculate FWHM
        i = profile_argmax
        while True:
            if sta_lta[i] <= profile_half:
                break
            i -= 1
            if i == 0:
                break
        fwhm_start_arg = i

        i = profile_argmax
        while True:
            if sta_lta[i] <= profile_half:
                break
            i += 1
            if i == len(sta_lta) - 1:
                break
        fwhm_end_arg = i

        profile_detail = {
            "profile_argmax": profile_argmax,
            "profile_max": profile_max,
            "profile_half": profile_half,
            "fwhm_start_arg": fwhm_start_arg,
            "fwhm_end_arg": fwhm_end_arg,
            "fwhm": fwhm_end_arg - fwhm_start_arg + 1,
        }
        profile_details.append(profile_detail)

    if plot_title or plot_filename:
        pprint(profile_details)

        fig = plt.figure(figsize=(24, 8))
        ax1 = fig.add_subplot(1, 1, 1)

        # plot STA/LTA
        ax1.set_ylabel("STA/LTA", color="blue")
        ax1.plot(sta_lta, color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.set_ylim(-1, 13)
        ax1.grid(which="major", axis="y", linestyle="dotted")

        # plot signal
        ax2 = ax1.twinx()
        ax2.set_ylabel("|m/s|", color="black")
        ax2.plot(
            pd.Series(stream[0].data[-len(sta_lta) :]).abs(), color="black", alpha=0.2
        )
        ax2.set_ylim(0, WAVEFORM_YLIM)

        # visualize argmax
        ax1.vlines(
            [profile_detail["profile_argmax"] for profile_detail in profile_details],
            0,
            np.max(sta_lta),
            color="red",
        )

        # visualize FWHM
        ax1.hlines(
            [profile["profile_half"] for profile in profile_details],
            [profile["fwhm_start_arg"] for profile in profile_details],
            [profile["fwhm_end_arg"] for profile in profile_details],
            color="#32CD32",
        )

        fig.tight_layout()
        ticks, datetime_ticks = get_datetime_ticks(stream)
        plt.xticks(ticks, datetime_ticks)
        if plot_title:
            plt.title(plot_title)
        if plot_filename:
            os.makedirs("/".join(plot_filename.split("/")[:-1]), exist_ok=True)
            plt.savefig(plot_filename)
        else:
            plt.show()

    return profile_details


def _calc_run_length(sequence):
    diff_seq = np.diff(sequence)

    newdata = np.append(True, diff_seq != 0)
    comp_seq = sequence[newdata]

    comp_seq_index = np.where(newdata)[0]
    comp_seq_index = np.append(comp_seq_index, len(sequence))
    lengths = np.diff(comp_seq_index)

    return comp_seq, lengths
