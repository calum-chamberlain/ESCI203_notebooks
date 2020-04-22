#!/usr/bin/env python3

"""
Interactive bokeh picker for earthquake waveforms

"""
import sys
import numpy as np
import datetime as dt
import warnings
import matplotlib.pyplot as plt

from typing import Iterable

from obspy import Stream, Inventory
from obspy.clients.fdsn import Client
from bokeh.plotting import figure, show
from bokeh.models import (
    HoverTool, ColumnDataSource, Legend, DataTable, TableColumn, PointDrawTool, 
    ColumnDataSource, DateFormatter)
from bokeh.models.formatters import DatetimeTickFormatter
from bokeh.layouts import gridplot, column, row


# Note Wood anderson sensitivity is 2080 as per Uhrhammer & Collins 1990
PAZ_WA = {'poles': [-6.283 + 4.7124j, -6.283 - 4.7124j],
          'zeros': [0 + 0j], 'gain': 1.0, 'sensitivity': 2080}


class Picker():
    """
    Earthquake picker.

    Parameters
    ----------
    
    """
    _response_removed = False
    _tools = ["pan", "box_zoom", "undo", "redo", "reset", "save", HoverTool(
        tooltips=[
            ("UTCDateTime", "@x{%m/%d %H:%M:%S.%3N}"),
            ("Amplitude", "@y")],
        formatters={'x': 'datetime'},
        mode='vline')]

    def __init__(
            self, 
            event_id: str = "2019p922847", 
            client: Client = Client("GEONET"),
            n_stations: int = 5,
            excluded_channels: Iterable = (
                "BNZ", "BNN", "BNE", "BN1", "BN2", "HNZ", "HNN", "HNE", "HN1",
                "HN2",),
            excluded_stations: Iterable = ("TUWZ"),
            all_components: bool = True,
            length: float = 60.,
            pre_pick: float = 10.,
            wood_anderson: bool = True,
        ):
        self.event_id = event_id
        self.client = client
        self.excluded_channels = excluded_channels
        self.excluded_stations = excluded_stations
        self.st, self._raw = Stream(), Stream()
        self.inventory = Inventory()
        self._get_st(
            n_stations=n_stations, all_components=all_components, 
            length=length, pre_pick=pre_pick)
        self._get_inventory()
        if wood_anderson:
            self.wood_anderson()

    def __repr__(self):
        return f"Picker(event_id={self.event_id})"

    def _get_st(
            self, 
            n_stations: int, 
            all_components: bool,
            length: float,
            pre_pick: float,
        ):
        event = self.client.get_events(eventid=self.event_id)[0]
        # sort p-picks by time
        p_picks = sorted(
            [p for p in event.picks if p.phase_hint.startswith("P") and 
             p.waveform_id.channel_code not in self.excluded_channels and
             p.waveform_id.station_code not in self.excluded_stations],
            key=lambda p: p.time)
        # Get a dictionary of the first n stations pick-times keyed by seed id
        seed_ids_to_download = {}
        for p_pick in p_picks:
            if len(seed_ids_to_download) == n_stations:
                break
            seed_id = p_pick.waveform_id.get_seed_string()
            if seed_id not in seed_ids_to_download.keys():
                seed_ids_to_download.update({seed_id: p_pick.time})
        if all_components:
            seed_ids_to_download = {
                key[0:-1] + "?": value 
                for key, value in seed_ids_to_download.items()}
        for seed_id, starttime in seed_ids_to_download.items():
            sys.stdout.write(f"Downloading data for {seed_id}    \r")
            sys.stdout.flush()
            try:
                self.st += self.client.get_waveforms(
                    *seed_id.split('.'), starttime=starttime - pre_pick,
                    endtime=(starttime - pre_pick) + length)
            except Exception as e:
                print(e)
        self.st = self.st.split().detrend().merge()
        self._raw = self.st.copy()  # Keep a safe raw version
        return

    def _get_inventory(self):
        bulk = [(tr.stats.network, tr.stats.station, tr.stats.location, 
                 tr.stats.channel, tr.stats.starttime, tr.stats.endtime)
                for tr in self.st]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # GeoNet doesn't pass xml version properly
            self.inventory += self.client.get_stations_bulk(bulk, level="response")
        return

    def remove_response(self, *args, **kwargs):
        self.st = self._raw.copy().remove_response(
            inventory=self.inventory, *args, **kwargs)

    def wood_anderson(self, water_level: float=10.):
        self.remove_response(output="VEL", water_level=water_level)
        self.st = self.st.simulate(
            paz_remove=None, paz_simulate=PAZ_WA, water_level=water_level)

    def plot(self, show: bool = True):
        fig = self.st.plot(show=False, handle=True)
        if show:
            plt.show()
        return fig

    def pick(
        self,         
        plot_height: int = 900, 
        plot_width: int = 1000
    ):
        trace_plots = self._define_plot()
        show(gridplot(trace_plots, ncols=1,
            # [[_trace_plot, _table] 
            #  for _trace_plot, _table in zip(trace_plots, tables)],
            plot_width=plot_width, plot_height=int(plot_width / len(self.st))))

    def _define_plot(
        self, 
        data_color: str = "darkblue",
    ):
        self.st.sort()
        min_time, max_time = (
            min(tr.stats.starttime for tr in self.st).datetime, 
            max(tr.stats.endtime for tr in self.st).datetime)
        # Define the data sources for plotting
        trace_sources = {}
        trace_data_range = {}
        # Allocate empty arrays
        for tr in self.st:
            times = np.arange(
                tr.stats.starttime.datetime,
                (tr.stats.endtime + tr.stats.delta).datetime,
                step=dt.timedelta(seconds=tr.stats.delta))
            data = tr.data
            trace_sources.update(
                {tr.id: {'time': times, 'data': data}})
            trace_data_range.update({tr.id: (data.min(), data.max())})

        plot_options=dict(
            tools=self._tools, x_axis_type="datetime", y_axis_location="right",
        )
        # Define an initial plot - attributes will be borrowed by other plots
        tr = self.st[0]
        p1 = figure(
             x_range=[min_time, max_time], **plot_options)
        p1.background_fill_color = "lightgrey"
        p1.yaxis.axis_label = None
        p1.xaxis.axis_label = None
        p1.min_border_bottom = 0
        p1.min_border_top = 0
        if len(self.st) != 1:
            p1.xaxis.major_label_text_font_size = '0pt'
        p1_line = p1.line(
            x=trace_sources[tr.id]["time"],
            y=trace_sources[tr.id]["data"],
            color=data_color, line_width=1)
        legend = Legend(items=[(tr.id, [p1_line])])
        p1.add_layout(legend, 'right')

        datetick_formatter = DatetimeTickFormatter(
            days=["%m/%d"], months=["%m/%d"],
            hours=["%m/%d %H:%M:%S"], minutes=["%m/%d %H:%M:%S"],
            seconds=["%m/%d %H:%M:%S"], hourmin=["%m/%d %H:%M:%S"],
            minsec=["%m/%d %H:%M:%S"])
        p1.xaxis.formatter = datetick_formatter

        # Add a picker tool
        # picks = ColumnDataSource({"x": [], "y": [], 'color': []})
        # renderer = p1.scatter(
        #         x='x', y='y', source=picks, color='color', size=10)
        # renderer_up = p1.ray(
        #     x='x', y='y', source=picks, color='color', angle=90, 
        #     angle_units="deg", length=0)
        # renderer_down = p1.ray(
        #     x='x', y='y', source=picks, color='color', angle=270, 
        #     angle_units="deg", length=0)
        # draw_tool = PointDrawTool(
        #     renderers=[renderer, renderer_up, renderer_down], 
        #     empty_value="red")
        # p1.add_tools(draw_tool)
        # p1.toolbar.active_tap = draw_tool

        # tabledatetimeformatter = DateFormatter(format="%m/%d %H:%M:%S.%3N")
        # columns = [TableColumn(field="x", title="Time", 
        #                        formatter=tabledatetimeformatter),
        #            TableColumn(field="y", title="Amplitude")]
        # table = DataTable(source=picks, columns=columns, editable=True)
        # all_picks = {tr.id: picks}
        trace_plots = [p1]
        # tables = [table]

        if len(self.st) <= 1:
            return
        for i, tr in enumerate(self.st[1:]):
            p = figure(x_range=p1.x_range, **plot_options)
            p.background_fill_color = "lightgrey"
            p.yaxis.axis_label = None
            p.xaxis.axis_label = None
            p.min_border_bottom = 0
            p.min_border_top = 0
            p_line = p.line(
                x=trace_sources[tr.id]["time"],
                y=trace_sources[tr.id]["data"],
                color=data_color, line_width=1)
            legend = Legend(items=[(tr.id, [p_line])])
            p.add_layout(legend, 'right')
            p.xaxis.formatter = datetick_formatter

            trace_plots.append(p)
            if i != len(self.st) - 2:
                p.xaxis.major_label_text_font_size = '0pt'
            
            # Add a picker tool
            # picks = ColumnDataSource({"x": [], "y": [], 'color': []})
            # renderer = p.scatter(
            #     x='x', y='y', source=picks, color='color', size=10)
            # renderer_up = p.ray(
            #     x='x', y='y', source=picks, color='color', angle=90, 
            #     angle_units="deg", length=0)
            # renderer_down = p.ray(
            #     x='x', y='y', source=picks, color='color', angle=270, 
            #     angle_units="deg", length=0)
            # draw_tool = PointDrawTool(
            #     renderers=[renderer, renderer_up, renderer_down], 
            #     empty_value="red")
            # p.add_tools(draw_tool)
            # p.toolbar.active_tap = draw_tool
            # all_picks.update({tr.id: picks})
            # columns = [TableColumn(field="x", title="Time",
            #                        formatter=tabledatetimeformatter),
            #            TableColumn(field="y", title="Amplitude")]
            # table = DataTable(source=picks, columns=columns, editable=True)
            # tables.append(table)

        return trace_plots #, tables


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bokeh picker applet")

    parser.add_argument(
        "-e", "--eventid", help="Event-ID for get from FDSN service",
        type=str, default="2019p922847")
    parser.add_argument(
        "-n", "--n-stations", type=int, default=5, 
        help="Number of stations to download")
    parser.add_argument(
        "-l", "--length", type=float, default=100., 
        help="Length in seconds to download and plot")

    args = parser.parse_args()

    Picker(
        event_id=args.eventid, n_stations=args.n_stations, 
        length=args.length).pick()