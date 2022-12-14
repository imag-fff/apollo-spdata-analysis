import os
from copy import deepcopy
from typing import Optional

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from numpy import ndarray
from obspy.core.stream import Stream
from scipy import signal

from src.constants import FONT_SIZE, WAVEFORM_YLIM
from src.utils import get_datetime_ticks

plt.rcParams["font.size"] = FONT_SIZE


def plot_current_data(
    stream, waveform_ylim: Optional[float] = None, filename: Optional[str] = None
) -> None:
    fig = plt.figure(figsize=(24, 4))
    ax = fig.add_subplot(1, 1, 1)
    if not waveform_ylim:
        waveform_ylim = WAVEFORM_YLIM

    ax.plot(stream[0].data, color="black")

    ax.set_ylim(-waveform_ylim, waveform_ylim)
    ax.set_ylabel("$m/s$")
    ax.grid(which="major")

    ticks, datetime_ticks = get_datetime_ticks(stream)
    ax.set_xticks(ticks, datetime_ticks)
    ax.set_title(f"Waveform ({stream[0].stats.station}, {stream[0].stats.channel})")

    if filename:
        os.makedirs("/".join(filename.split("/")[:-1]), exist_ok=True)
        plt.savefig(filename)
    else:
        plt.show()


def plot_spectrogram(
    stream: Stream,
    waveform_ylim: Optional[float] = None,
    filename: Optional[str] = None,
) -> None:
    # initial settings
    fig = plt.figure(figsize=(24, 16))
    plt.gca().spines[:].set_visible(False)
    plt.axis("off")
    ticks, datetime_ticks = get_datetime_ticks(stream)
    if not waveform_ylim:
        waveform_ylim = WAVEFORM_YLIM

    # plot waveform
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(stream[0].data)
    ax.set_xticks(ticks, [None] * len(ticks))
    ax.set_xmargin(0)
    ax.set_ylim(-waveform_ylim, waveform_ylim)
    ax.set_title("Waveform")
    ax.set_ylabel("$m/s$")
    ax.grid(which="major")

    # plot spectrogram
    N_SECONDS = 50
    N_SAMPLES = int(stream[0].stats.sampling_rate * N_SECONDS)
    ax = fig.add_subplot(2, 1, 2)
    f, t, sxx = signal.spectrogram(
        stream[0].data * 1e9,
        fs=stream[0].stats.sampling_rate,
        window="hamming",
        nperseg=N_SAMPLES,
        noverlap=N_SAMPLES // 2,
        nfft=N_SAMPLES,
    )
    plt.pcolormesh(
        t,
        f,
        np.sqrt(sxx),
        cmap=plt.get_cmap("jet"),
        norm=colors.LogNorm(vmin=0.2, vmax=100),
    )
    plt.title("Spectrogram (High Freq)")

    SEP = 8
    n_t = len(t)
    t_ticks = [i * n_t // SEP for i in range(SEP)] + [n_t - 1]

    ax.set_xticks(t[t_ticks], datetime_ticks)
    ax.set_xlabel("Datetime (UTC)")

    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(1, 16)
    ax.set_yscale("log")
    ax.set_yticks([2, 4, 8])
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    ax.get_yaxis().set_tick_params(which="minor", size=0)

    # other settings
    plt.colorbar(orientation="horizontal", label="PSD ($nm/s/???Hz$)", aspect=80)
    plt.grid(which="major")
    plt.subplots_adjust(hspace=0.15)

    if filename:
        os.makedirs("/".join(filename.split("/")[:-1]), exist_ok=True)
        plt.savefig(filename)
    else:
        plt.show()


def plot_event_detection(
    stream: Stream,
    sta_lta: ndarray,
    start_args: ndarray,
    sta_lta_threshold: float,
    ts_sec: float,
    waveform_ylim: Optional[float] = None,
    filename: Optional[str] = None,
    logging_path: Optional[str] = None,
) -> None:
    # initial settings
    fig = plt.figure(figsize=(24, 24))
    plt.gca().spines[:].set_visible(False)
    plt.axis("off")
    sta_lta_offset = len(stream[0].data) - len(sta_lta)
    ticks, datetime_ticks = get_datetime_ticks(stream)
    if not waveform_ylim:
        waveform_ylim = WAVEFORM_YLIM

    # plot STA/LTA
    sta_lta_over_threshold = deepcopy(sta_lta)
    sta_lta_over_threshold[sta_lta < sta_lta_threshold] = None
    ax = fig.add_subplot(3, 1, 1)
    ax.set_ylabel("STA/LTA")
    ax.plot(np.append([None] * sta_lta_offset, sta_lta), color="black")
    ax.plot(
        np.append([None] * sta_lta_offset, sta_lta_over_threshold),
        color="red",
    )
    ax.plot(np.repeat(sta_lta_threshold, len(sta_lta)), color="#32CD32")
    ax.set_xticks(ticks, [None] * len(ticks))
    ax.set_xmargin(0)
    ax.set_ylim(-1, 13)
    ax.set_title("Event trigger")
    ax.grid(which="major")

    # plot waveform
    ax = fig.add_subplot(3, 1, 2)
    ax.plot(stream[0].data, color="black")
    ax.vlines(
        start_args + sta_lta_offset - round(ts_sec * stream[0].stats.sampling_rate),
        -waveform_ylim,
        waveform_ylim,
        color="red",
    )
    ax.set_xticks(ticks, [None] * len(ticks))
    ax.set_xmargin(0)
    ax.set_ylim(-waveform_ylim, waveform_ylim)
    ax.set_title("Waveform")
    ax.set_ylabel("$m/s$")
    ax.grid(which="major")

    # plot spectrogram
    N_SECONDS = 50
    N_SAMPLES = int(stream[0].stats.sampling_rate * N_SECONDS)
    ax = fig.add_subplot(3, 1, 3)
    f, t, sxx = signal.spectrogram(
        stream[0].data * 1e9,
        fs=stream[0].stats.sampling_rate,
        window="hamming",
        nperseg=N_SAMPLES,
        noverlap=N_SAMPLES // 2,
        nfft=N_SAMPLES,
    )
    plt.pcolormesh(
        t,
        f,
        np.sqrt(sxx),
        cmap=plt.get_cmap("jet"),
        norm=colors.LogNorm(vmin=0.2, vmax=100),
    )
    plt.title("Spectrogram (High Freq)")

    SEP = 8
    n_t = len(t)
    t_ticks = [i * n_t // SEP for i in range(SEP)] + [n_t - 1]

    ax.set_xticks(t[t_ticks], datetime_ticks)
    ax.set_xlabel("Datetime (UTC)")

    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(1, 16)
    ax.set_yscale("log")
    ax.set_yticks([2, 4, 8])
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    ax.get_yaxis().set_tick_params(which="minor", size=0)

    # other settings
    plt.colorbar(orientation="horizontal", label="PSD ($nm/s/???Hz$)", aspect=80)
    plt.grid(which="major")
    plt.subplots_adjust(hspace=0.15)

    if logging_path:
        with open(logging_path, "a") as log:
            for start_arg in start_args:
                print(
                    stream[0].stats.starttime + stream[0].times()[start_arg], file=log
                )

    if filename:
        os.makedirs("/".join(filename.split("/")[:-1]), exist_ok=True)
        plt.savefig(filename)
    else:
        plt.show()
