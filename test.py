import pickle
from datetime import datetime, timedelta

from obspy.core.utcdatetime import UTCDateTime

from src.get_seismogram import get_seismogram_iris
from src.preprocess import reduce_noise, remove_response

for station in ["S14", "S15", "S16"]:
    starttime = UTCDateTime(1971, 1, 1) - timedelta(days=1)
    endtime = starttime + timedelta(days=1)

    while True:
        with open("logs/log_get_all.txt", "a") as f:
            try:
                starttime = starttime + timedelta(hours=12)
                endtime = endtime + timedelta(hours=12)

                filename = station + "_" + str(starttime)[:13] + ".pkl"
                print(f"{filename} => ", end="", file=f)

                # 1_raw
                st = get_seismogram_iris(
                    starttime=starttime,
                    endtime=endtime,
                    station=station,
                )
                if st is None:
                    print("ERROR", file=f)
                    continue
                with open(
                    f"/Volumes/BUFFALO_5TB/pickles/SHZ/1_raw/{filename}", "wb"
                ) as p:
                    pickle.dump(st, p)
                print("1_raw => ", end="", file=f)

                # 2_remove_response
                st = remove_response(
                    stream=st,
                    starttime=starttime,
                    endtime=endtime,
                    station=station,
                )
                if st is None:
                    print("ERROR", file=f)
                    continue
                with open(
                    f"/Volumes/BUFFALO_5TB/pickles/SHZ/2_remove_response/{filename}",
                    "wb",
                ) as p:
                    pickle.dump(st, p)
                print("2_remove_response => ", end="", file=f)

                # 3_reduce_noise
                st = reduce_noise(st)
                if st is None:
                    print("ERROR", file=f)
                    continue
                with open(
                    f"/Volumes/BUFFALO_5TB/pickles/SHZ/3_reduce_noise/{filename}", "wb"
                ) as p:
                    pickle.dump(st, p)
                print("3_reduce_noise", file=f)

            except Exception as e:
                print("ERROR", file=f)
                print(e)

        if starttime >= datetime(1977, 10, 1):
            break
