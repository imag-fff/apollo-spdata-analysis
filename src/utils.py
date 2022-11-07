from obspy.core.stream import Stream


def get_datetime_ticks(stream: Stream, sep: int = 8) -> tuple[list, list]:
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
