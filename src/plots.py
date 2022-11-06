import os
from copy import deepcopy
from typing import Optional

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from obspy.core.stream import Stream
from scipy import signal

plt.rcParams["font.size"] = 18


def _get_datetime_ticks(stream: Stream, sep: int = 8):
    starttime = stream[0].stats.starttime
    datetimes = stream[0].times()
    npts = stream[0].stats.npts

    ticks = [i * npts // sep for i in range(sep)] + [npts - 1]
    datetime_ticks = [starttime + datetimes[tick] for tick in ticks]
    datetime_ticks = [
        " ".join(str(item).split(".")[0].split("T")) if i % 2 == 0 else ""
        for i, item in enumerate(datetime_ticks)
    ]

    return ticks, datetime_ticks


def plot_spectrogram(stream: Stream, filename: Optional[str] = None):
    # initial settings
    fig = plt.figure(figsize=(24, 12))
    plt.gca().spines[:].set_visible(False)
    plt.axis("off")
    ticks, datetime_ticks = _get_datetime_ticks(stream)

    # plot waveform
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(stream[0].data)
    ax.set_xticks(ticks, [None] * len(ticks))
    ax.set_xmargin(0)
    ax.set_ylim(-3e-7, 3e-7)
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
        norm=colors.LogNorm(vmin=0.3, vmax=100),
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
    plt.colorbar(orientation="horizontal", label="PSD ($nm/s/√Hz$)", aspect=80)
    plt.grid(which="major")
    plt.subplots_adjust(hspace=0.1)

    if filename:
        os.makedirs("".join(filename.split("/")[:-1]), exist_ok=True)
        plt.savefig(filename)
    else:
        plt.show()
