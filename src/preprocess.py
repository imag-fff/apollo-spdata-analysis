from copy import deepcopy
from enum import Enum
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from obspy.clients.fdsn.client import Client
from obspy.core import UTCDateTime
from obspy.core.stream import Stream
from obspy.core.utcdatetime import UTCDateTime
from scipy import signal

from src.pdart_utils import linear_interpolation
from src.utils import get_datetime_ticks

plt.rcParams["font.size"] = 24


def remove_response(
    stream: Stream,
    starttime: Union[UTCDateTime, list[int]],
    endtime: Union[UTCDateTime, list[int]],
    network: str = "XA",
    station: str = "S14",
    channel: str = "SHZ",
    location: str = "*",
    is_filled: bool = True,
    inplace: bool = False,
) -> Optional[Stream]:
    try:
        st = stream.copy() if not inplace else stream
        client = Client("IRIS")
        start = UTCDateTime(*starttime) if type(starttime) == list else starttime
        end = UTCDateTime(*endtime) if type(endtime) == list else endtime

        inv = client.get_stations(
            starttime=start,
            endtime=end,
            network=network,
            sta=station,
            loc=location,
            channel=channel,
            level="response",
        )
        for tr in st:
            if tr.stats.channel in ["MH1", "MH2", "MHZ"]:
                PRE_FILT = [0.1, 0.3, 0.9, 1.1]
            elif tr.stats.channel in ["SHZ"]:
                PRE_FILT = [1, 2, 11, 13]

            original_mask = linear_interpolation(tr, interpolation_limit=None)
            tr.remove_response(
                inventory=inv, pre_filt=PRE_FILT, output="VEL", water_level=None
            )  # output is one of ["DISP", "VEL", "ACC", "DEF"]
            tr.data = np.ma.masked_array(tr, mask=original_mask)

            if is_filled:
                tr.data = tr.data.filled(0)

        if not inplace:
            return st

    except Exception as e:
        print(f"========== ERROR ==========\n\n{e}\n")
        return


class NoiseReductionMethod(Enum):
    ENVELOPE = "envelope"
    EWM = "ewm"


def reduce_noise(
    stream: Stream,
    method: NoiseReductionMethod = NoiseReductionMethod.ENVELOPE.value,
    times: int = 5,
    inplace: bool = False,
    is_plot: bool = False,
) -> Stream:
    st = stream.copy() if not inplace else stream
    d = st[0].data
    if is_plot:
        ticks, datetime_ticks = get_datetime_ticks(st)

        def _plot(
            data_prev: Optional[ndarray] = None,
            data_current: Optional[ndarray] = None,
            threshold: Optional[ndarray] = None,
            title: Optional[str] = None,
        ) -> None:
            fig = plt.figure(figsize=(24, 8))
            ax = fig.add_subplot(1, 1, 1)

            if data_prev is not None:
                ax.plot(data_prev, color="red", label="previous data")
            if data_current is not None:
                ax.plot(data_current, color="black", label="current data")
            if threshold is not None:
                ax.plot(threshold, color="#32CD32", label="threshold")
                ax.plot(-threshold, color="#32CD32")
            if title:
                ax.set_title(title)

            ax.set_xticks(ticks, datetime_ticks)
            ax.set_xlabel("Datetime (UTC)")
            ax.set_ylim(-3e-7, 3e-7)
            ax.set_ylabel("$m/s$")
            ax.grid(which="major")
            ax.legend(loc="upper right")
            plt.show()

        _plot(data_current=d, title="Before")

    if method == NoiseReductionMethod.ENVELOPE.value:
        # hyperparameters
        N_SECONDS = 60
        N_SAMPLES = int(st[0].stats.sampling_rate * N_SECONDS)  # window size
        ENV_MUL, STD_MUL = 4, 8

        for t in range(times):
            if is_plot:
                d_before = deepcopy(d)

            threshold = (
                np.convolve(
                    np.ones(N_SAMPLES) / N_SAMPLES, abs(signal.hilbert(d)), mode="same"
                )
                * ENV_MUL
                + np.var(d) * STD_MUL
            )
            d[np.abs(d) > threshold] = 0

            if is_plot:
                _plot(
                    d_before,
                    d,
                    threshold,
                    title=f"Noise reduction with {method} ({t+1} times)",
                )

    elif method == NoiseReductionMethod.EWM.value:
        # hyperparameters
        N_SECONDS = 20
        N_SAMPLES = int(st[0].stats.sampling_rate * N_SECONDS)  # window size
        STD_MUL = 4

        d = np.append(d, np.zeros(N_SAMPLES))  # add dummy data
        d = pd.Series(d)
        for t in range(times):
            if is_plot:
                d_before = deepcopy(d)

            threshold = (d.ewm(span=N_SAMPLES).std() * STD_MUL)[N_SAMPLES:].reset_index(
                drop=True
            )
            d[:-N_SAMPLES][d.abs()[:-N_SAMPLES] > threshold] = 0

            if is_plot:
                _plot(
                    d_before,
                    d,
                    threshold,
                    title=f"Noise reduction with {method} ({t+1} times)",
                )

        d = np.array(d)
        d = d[:-N_SAMPLES]  # remove dummy data

    st[0].data = d
    if is_plot:
        _plot(data_current=d, title="After")

    if not inplace:
        return st
