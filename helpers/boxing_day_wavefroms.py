#!/usr/env/bin python3

"""
Script to make a waveform viewer for the Boxing Day earthquake.

"""

from bokeh_picker import BokehWaveformViewer

from obspy import UTCDateTime
from obspy.clients.fdsn import Client

STATIONS = [
    ("ANTO", "BHZ"), ("BFO", "BHZ"), ("CASY", "BHZ"), ("CHTO", "BHZ"),
    ("COCO", "BHZ"), ("DAV", "BHZ"), ("DGAR", "BHZ"), ("FURI", "BHZ"),
    ("GNI", "BHZ"), ("GUMO", "BHZ"), ("KMBO", "BHZ"), ("LSA", "BHZ"),
    ("MBWA", "10", "BHZ"), ("NWAO", "BHZ"), ("PALK", "BHZ"), ("PMG", "BHZ"),
    ("QIZ", "BHZ"), ("RAYN", "BHZ"), ("SNZO", "BHZ"), ("ULN", "BHZ"),
    ("WRAB", "BHZ"), ("YSS", "BHZ")]
STARTTIME = UTCDateTime(2004, 12, 26, 0, 58, 53)
LENGTH = 850.0

def get_st(lowcut: float = None, highcut: float = None):
    client = Client("IRIS")
    bulk = [("I?", sta[0], "00", sta[1], STARTTIME, STARTTIME + LENGTH) 
            for sta in STATIONS]
    st = client.get_waveforms_bulk(bulk)
    st.merge().detrend().resample(20.0)
    if lowcut and highcut:
        st.filter("bandpass", freqmin=lowcut, freqmax=highcut, corners=4)
    st.sort(["network", "station", "location", "channel"])
    return st


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bxing Day earthquake viewer")
    parser.add_argument(
        "-o", "--output", type=str, required=False, 
        help="File output to, otherwise, will show")
    args = parser.parse_args()
    st = get_st(1, 5)
    BokehWaveformViewer(st=st, amplitude_units="Counts").pick(
        plot_height=1000, output=args.output)
