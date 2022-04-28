"""
Functions to create an animation to explain focal mechanisms
"""

import numpy as np
import progressbar

from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel
from obspy.taup.tau import COLORS
from obspy.geodetics import locations2degrees

from obspy.core.event import Origin
from obspy.core.inventory import Station

import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrow

import cartopy.crs as ccrs
import cartopy.feature as cfeature


MODEL = TauPyModel(model="iasp91")  # Change this to use a different velocity model
DEFAULT_PHASES = ["ttbasic"]
POLARIZATION_CMAP = cm.get_cmap("coolwarm", 1000)
FPS = 10


def map_plot(fig: Figure, source: Origin, reciever: Station, position=221, 
             resolution="110m"):
    """ Plot a map with a great-circle between source and reciever """
    source_lon, source_lat = source.longitude % 360, source.latitude
    reciever_lon, reciever_lat = reciever.longitude % 360, reciever.latitude

    proj = ccrs.Mollweide(central_longitude=(source_lon + reciever_lon) / 2)
    ax = fig.add_subplot(position, projection=proj)
    ax.set_global()
    borders = cfeature.NaturalEarthFeature(
         cfeature.BORDERS.category, cfeature.BORDERS.name, resolution,
         edgecolor='none', facecolor='none')
    land = cfeature.NaturalEarthFeature(cfeature.LAND.category,
                                        cfeature.LAND.name, resolution,
                                        edgecolor='face', facecolor='none')
    ocean = cfeature.NaturalEarthFeature(cfeature.OCEAN.category,
                                         cfeature.OCEAN.name, resolution,
                                         edgecolor='face',
                                         facecolor='none')
    # ax.set_axis_bgcolor('1.0')
    ax.add_feature(ocean, facecolor='1.0')
    ax.add_feature(land, facecolor='0.8')
    ax.add_feature(borders, edgecolor='0.75')
    ax.coastlines(resolution=resolution, color='0.4')
    ax.gridlines(xlocs=range(-180, 181, 30), ylocs=range(-90, 91, 30))

    source_scatter = ax.scatter(
        source_lon, source_lat, marker="o", s=100, c="red", zorder=10, 
        transform=ccrs.Geodetic(), label="Source")
    reciever_scatter = ax.scatter(
        reciever_lon, reciever_lat, marker="v", s=100, c="blue", zorder=10, 
        transform=ccrs.Geodetic(), label="Receiver")
    # Plot great circle
    ax.plot([source_lon, reciever_lon], [source_lat, reciever_lat], "k",
            transform=ccrs.Geodetic())
    ax.legend()
    
    return fig, ax


def ray_plot(fig: Figure, source: Origin, reciever: Station, 
             phases: list = None, position=223):
    """ Use TauP to make a plot of predicted ray-paths. """
    phases = phases or DEFAULT_PHASES
    ax = fig.add_subplot(position, projection="polar")

    degrees_dist = locations2degrees(
        reciever.latitude, reciever.longitude, source.latitude, 
        source.longitude)
    arrivals = MODEL.get_ray_paths(
        source_depth_in_km=source.depth / 1000.0, 
        distance_in_degree=degrees_dist, phase_list=phases)
    arrivals.plot_rays(ax=ax, legend=True, show=False)
    return fig, ax


def polarisation_plot(fig, tr_z, tr_n, tr_e, position=222):
    """ Make a graph of wave polarisation. Animated. """
    ax = fig.add_subplot(position)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("North")
    ax.set_xlabel("East")
    ax.yaxis.set_label_position("left")
    ax.xaxis.set_label_position("top")

    mappable = cm.ScalarMappable(norm=Normalize(0, 1), cmap=POLARIZATION_CMAP)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["Down", "Up"])

    return fig, ax


def waveform_plot(fig, tr_z, tr_n, tr_e, source: Origin, reciever: Station, 
                  phases: list = None,position=224):
    """ Waveform plot - animated. """
    ax = fig.add_subplot(position)
    # Normalise to maximum amplitude across traces
    max_amp = max(tr_z.data.max(), tr_e.data.max(), tr_n.data.max())
    assert tr_z.stats.starttime == tr_e.stats.starttime == tr_n.stats.starttime
    assert tr_z.stats.npts == tr_n.stats.npts == tr_e.stats.npts
    times = [(tr_z.stats.starttime + t).datetime 
             for t in (np.arange(tr_z.stats.npts) / tr_z.stats.sampling_rate)]
    z, n, e = tr_z.data / max_amp, tr_n.data / max_amp, tr_e.data / max_amp
    # Offset each trace by that max amplitude and plot - turn off y-ticks
    n += e.max() + abs(n.min())
    z += n.max() + abs(z.min())
    # Plot in grey first, then animate in black over the top.
    ax.plot(times, e, color="lightgrey")
    ax.plot(times, n, color="lightgrey")
    ax.plot(times, z, color="lightgrey")
    ax.set_yticks([])
    ax.text(x=times[0], y=e.mean() + 0.2, s=tr_e.id)
    ax.text(x=times[0], y=n.mean() + 0.2, s=tr_n.id)
    ax.text(x=times[0], y=z.mean() + 0.2, s=tr_z.id)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    # Plot arrivals
    phases = phases or DEFAULT_PHASES
    degrees_dist = locations2degrees(
        reciever.latitude, reciever.longitude, source.latitude, 
        source.longitude)
    arrivals = MODEL.get_travel_times(
        source_depth_in_km=source.depth / 1000.0, 
        distance_in_degree=degrees_dist, phase_list=phases)
    phase_list_plotted = []
    i = 0
    for arrival in arrivals:
        arrival_time = source.time + arrival.time
        if arrival_time < tr_z.stats.endtime:
            ax.axvline(arrival_time.datetime, 
            color=COLORS[i % len(COLORS)], label=arrival.name)
            phase_list_plotted.append(arrival.name)
            i += 1
    ax.legend()
    return fig, ax, phase_list_plotted


def define_plot(origin, station, st):
    tr_z = st.select(component="Z")[0]
    tr_n = (st.select(component="N") or st.select(component="1"))[0]
    tr_e = (st.select(component="E") or st.select(component="2"))[0]

    fig = plt.figure(figsize=[15, 10])

    fig, pol_ax = polarisation_plot(
        fig=fig, tr_z=tr_z, tr_n=tr_n, tr_e=tr_e)
    fig, wav_ax, phase_list_plotted = waveform_plot(
        fig=fig, tr_z=tr_z, tr_e=tr_e, tr_n=tr_n, source=origin, 
        reciever=station, phases=DEFAULT_PHASES)

    fig, map_ax = map_plot(fig=fig, source=origin, reciever=station)
    fig, ray_ax = ray_plot(fig=fig, source=origin, reciever=station, 
                           phases=phase_list_plotted)
    
    anim = animate_figure(fig, pol_ax, wav_ax, tr_z, tr_n, tr_e)
    return anim


def animate_figure(fig, pol_ax, wav_ax, tr_z, tr_n, tr_e):
    print("Starting animation")
    times = [(tr_z.stats.starttime + t).datetime 
             for t in (np.arange(tr_z.stats.npts) / tr_z.stats.sampling_rate)]
        # Plot vector
    max_amp = max(tr_z.data.max(), tr_e.data.max(), tr_n.data.max())
    z_norm, n_norm, e_norm = (
        tr_z.data / max_amp, tr_n.data / max_amp, tr_e.data / max_amp)
    n_norm += e_norm.max() + abs(n_norm.min())
    z_norm += n_norm.max() + abs(z_norm.min())
    _min_amp = e_norm.min()
    _max_amp = z_norm.max()

    tr_z.detrend()
    z = tr_z.data + abs(tr_z.data.min())
    z /= z.max()
    # For plotting this needs to be between 0 and 1
    n = tr_n.data / np.abs(tr_n.data).max()
    e = tr_e.data / np.abs(tr_e.data).max()

    pol_arrow = FancyArrow(
        0, 0, 0, 0, facecolor=POLARIZATION_CMAP(0), width=0.1, lw=1, ls="-", 
        edgecolor='k')
    pol_ax.add_patch(pol_arrow)

    z_line, = wav_ax.plot([], [], "k", lw=2)
    n_line, = wav_ax.plot([], [], "k", lw=2)
    e_line, = wav_ax.plot([], [], "k", lw=2)
    vline, = wav_ax.plot([], [], "k--", lw=1)

    bar = progressbar.ProgressBar(max_value=len(tr_z.data))
    def update(i):
        # Update the lines attributes
        z_line.set_data(times[0:i], z_norm[0:i])
        n_line.set_data(times[0:i], n_norm[0:i])
        e_line.set_data(times[0:i], e_norm[0:i])
        vline.set_data([times[i], times[i]], [_min_amp, _max_amp])
        bar.update(i)

        pol_ax.patches.pop(0)
        pol_arrow = FancyArrow(
            0, 0, e[i], n[i], facecolor=POLARIZATION_CMAP(z[i]), width=0.1, 
            lw=1, ls="-", edgecolor='k')
        pol_ax.add_patch(pol_arrow)
        return z_line, n_line, e_line, pol_arrow, vline


    anim = FuncAnimation(fig, update, range(len(tr_z.data)),
                         blit=True, interval=1000/FPS, repeat=False)
    return anim


def main(clientid: str, eventid: str, stationid: str, location_code: str,
         duration: float):
    client = Client(clientid)
    try:
        event = client.get_events(eventid=eventid)[0]
    except:
        raise FileNotFoundError(f"Event id not recognised {eventid}")
    origin = event.preferred_origin() or event.origins[0]
    station = client.get_stations(
        station=stationid, location=location_code, 
        endafter=origin.time, startbefore=origin.time)[0][0]
    # Predict arrival times using TauP
    degrees_dist = locations2degrees(
        station.latitude, station.longitude, origin.latitude, origin.longitude)
    p_arrival = origin.time + MODEL.get_travel_times(
        source_depth_in_km=origin.depth / 1000.0, 
        distance_in_degree=degrees_dist, phase_list=["P"])[0].time
    # Get waveforms around that time
    st = client.get_waveforms(
        station=stationid, location=location_code, starttime=p_arrival - 30,
        network="*", channel="BH?",  endtime=p_arrival + duration)
    return define_plot(origin=origin, station=station, st=st)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-panel animation of wave polarisation, uses an FDSN "
                    "client")

    parser.add_argument(
        "-e", "--eventid", help="Event-ID to download", type=str,
        default="3279407")  # Tohoku-Oki
    parser.add_argument(
        "-c", "--clientid", type=str, default="IRIS",
        help="Client ID to use, must be FDSN with waveform and event services")
    parser.add_argument(
        "-s", "--station", type=str, default="SNZO",
        help="Station name to download waveforms for")
    parser.add_argument(
        "-l", "--location-code", type=str, default="10",
        help="Location code for instrument to download")
    parser.add_argument(
        "-d", "--duration", type=float, default=1200,
        help="Duration in seconds to download after P-arrival")
    parser.add_argument(
        "-o", "--outfile", type=str, required=False,
        help="Outfile to save to, if not set, will show to screen")
    
    args = parser.parse_args()
    anim = main(clientid=args.clientid, eventid=args.eventid,
                stationid=args.station, location_code=args.location_code,
                duration=args.duration)
    if args.outfile:
        anim.save(args.outfile, writer="ffmpeg", codec="h264")
    else:
        plt.show()