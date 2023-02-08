import pickle
from glob import glob
from time import sleep

from src.detect_events import calc_sta_lta, get_args_over_sta_lta_threshold
from src.plots import plot_event_detection

files = sorted(glob("/Volumes/BUFFALO_5TB/pickles/SHZ/3_reduce_noise/*"))

TL_SEC, TS_SEC, THRESHOLD = 2400, 600, 1.5
FILENAME_PREFIX = "/Users/imagawa/Desktop/実験データ/Event_Detection/"

for file in files[3109:]:
    with open("./logs/event_detection.txt", "a") as f:
        try:
            filename = FILENAME_PREFIX + (file[:-3] + "png").split("/")[-1]
            print(f"{filename} => ", end="", file=f)

            with open(file, "rb") as p:
                st = pickle.load(p)

            sta_lta = calc_sta_lta(stream=st, tl_sec=TL_SEC, ts_sec=TS_SEC)

            start_args, end_args = get_args_over_sta_lta_threshold(
                stream=st,
                input_sta_lta=sta_lta,
                sta_lta_threshold=THRESHOLD,
                run_length_threshold=10,
            )
            plot_event_detection(
                stream=st,
                sta_lta=sta_lta,
                start_args=start_args,
                sta_lta_threshold=THRESHOLD,
                ts_sec=TS_SEC,
                waveform_ylim=1.2e-7,
                filename=filename,
                logging_path="./logs/starttimes.txt",
            )

            with open("./logs/starttimes.txt", "a") as log:
                print(file=log)

            print("OK", file=f)

        except Exception as e:
            print(e)
            print("ERROR", file=f)
            sleep(5)
