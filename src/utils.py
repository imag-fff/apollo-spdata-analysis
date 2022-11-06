from copy import deepcopy
from typing import Optional

import numpy as np
from obspy.clients.fdsn.client import Client
from obspy.core.stream import Stream
from obspy.core.utcdatetime import UTCDateTime

from src.pdart_utils import linear_interpolation


def remove_response(
    stream: Stream,
    starttime: list[int],
    endtime: list[int],
    network: str = "XA",
    station: str = "S14",
    channel: str = "SHZ",
    location: str = "*",
    is_filled: bool = True,
    inplace=False,
) -> Optional[Stream]:
    try:
        st = stream.copy() if not inplace else stream
        client = Client("IRIS")
        starttime, endtime = UTCDateTime(*starttime), UTCDateTime(*endtime)

        inv = client.get_stations(
            starttime=starttime,
            endtime=endtime,
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
        print(f"========== ERROR ==========\n\n{e}")
        return
