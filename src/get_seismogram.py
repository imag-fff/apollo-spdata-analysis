from typing import Optional, Union

from obspy.clients.fdsn.client import Client
from obspy.core.stream import Stream
from obspy.core.utcdatetime import UTCDateTime

from src.pdart_utils import linear_interpolation


def get_seismogram_iris(
    starttime: Union[UTCDateTime, list[int]],
    endtime: Union[UTCDateTime, list[int]],
    network: str = "XA",
    station: str = "S14",
    channel: str = "SHZ",
    location: str = "*",
) -> Optional[Stream]:
    try:
        client = Client("IRIS")

        start = UTCDateTime(*starttime) if type(starttime) == list else starttime
        end = UTCDateTime(*endtime) if type(endtime) == list else endtime

        st = client.get_waveforms(
            starttime=start,
            endtime=end,
            network=network,
            station=station,
            channel=channel,
            location=location,
        )
        for tr in st:
            linear_interpolation(tr, interpolation_limit=1)
        st.merge()

        return st

    except Exception as e:
        print(f"========== ERROR ==========\n\n{e}\n")
        return
