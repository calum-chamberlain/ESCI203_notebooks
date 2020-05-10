"""
Simple viewer for GPS data from GeoNet.

"""

import requests
from datetime import datetime as dt
import numpy as np

from bokeh.plotting import figure, show, output_file, save
from bokeh.models import (
    HoverTool, DateFormatter, Band, ColumnDataSource, DataTable, TableColumn, 
    PointDrawTool, NumberFormatter, Div)
from bokeh.models.formatters import DatetimeTickFormatter
from bokeh.layouts import gridplot, column, row


GEONET_FITS = "http://fits.geonet.org.nz/observation"


class GPSData():
    def __init__(self, reciever: str, component: str, times: np.ndarray, 
                 observations: np.ndarray, errors: np.ndarray):
        assert times.shape == observations.shape, "Times is not shaped like observations"
        assert observations.shape == errors.shape, "Observations is not shaped like errors"
        self.reciever = reciever
        self.component = component
        # Sort everything
        sort_mask = np.argsort(times)
        self.times = times[sort_mask]
        self.observations = observations[sort_mask]
        self.errors = errors[sort_mask]

    def trim(self, starttime: dt = None, endtime: dt = None):
        starttime = starttime or self.times[0]
        endtime = endtime or self.times[-1]
        mask = np.where(
            np.logical_and(self.times >= starttime, self.times <= endtime))
        self.times = self.times[mask]
        self.observations = self.observations[mask]
        self.errors = self.errors[mask]
        return self


class GPSViewer():
    components = ("up", "north", "east")
    def __init__(self, reciever: str, starttime: dt, endtime: dt): 
        self.reciever = reciever
        self.starttime = starttime
        self.endtime = endtime
        self.get_data()

    def get_data(self):
        """ Get data from the GeoNet FITS service. """
        for channel in self.components:
            parameters = {"typeID": channel[0], "siteID": self.reciever}
            response = requests.get(GEONET_FITS, params=parameters)
            assert response.status_code == 200, "Bad request"
            payload = response.content.decode("utf-8").split("\n")
            # payload is a csv with header
            # This is a list-comprehension, a type of fast, one-line for loop
            payload = [p.split(',') for p in payload]
            # Check that this is what we expect
            assert payload[0][0] == 'date-time', "Unkown format"
            assert len(payload[0]) == 3, "Unknown format"
            times, displacements, errors = zip(*[
                (dt.strptime(p[0], '%Y-%m-%dT%H:%M:%S.%fZ'),
                float(p[1]), float(p[2])) for p in payload[1:-1]])
            self.__dict__.update({channel: GPSData(
                reciever=self.reciever, component=channel, times=np.array(times),
                observations=np.array(displacements), 
                errors=np.array(errors)).trim(
                    starttime=self.starttime, endtime=self.endtime)})

    def plot(self, method: str = "bokeh", output = None, *args, **kwargs):
        """ Plot data """
        assert method.lower() in ("bokeh", "matplotlib", "mpl"), "Only Matplotlib or bokeh supported"
        if method.lower() == "bokeh":
            self._bokeh_plot(output=output, *args, **kwargs)
        else:
            self._mpl_plot(output=output, *args, **kwargs)

    def _bokeh_plot(
        self, 
        plot_height: int = 1000, 
        plot_width: int = 1000,
        output = None,
        *args, **kwargs):
        """ Make an interactive bokeh plot. """
        core_axis = None
        plots = [Div(text=f"<b>GPS displacements for {self.reciever}</b>",
                     style={'font-size': "200%"})]
        tools =  ["pan", "box_zoom", "reset", "save", "crosshair"]
        plot_options = dict(
            tools=tools, x_axis_type="datetime", y_axis_location="right",
            plot_width=plot_width, 
            plot_height=int(plot_height / len(self.components)))
        for component in self.components:
            plot, table = self._bokeh_component_plot(
                component=component, core_axis=core_axis, 
                plot_options=plot_options)
            if core_axis is None:
                core_axis = plot
            plots.append(row(plot, table))
        layout = column(plots, sizing_mode="scale_both")
        if output:
            output_file(output, title="GPS Viewer")
            save(layout)
        else:
            show(layout)
        pass

    def _bokeh_component_plot(
        self, 
        component: str, 
        plot_options: dict, 
        data_color: str = "darkblue", 
        core_axis = None
    ):
        """ Define a plot for an individual component. """
        times, obs, err = (
            self.__dict__[component].times, 
            self.__dict__[component].observations,
            self.__dict__[component].errors)
        source = ColumnDataSource(dict(
            times=times, obs=obs, err=err, lower=obs - err, upper=obs + err))
        
        if not core_axis:
            p = figure(title=component, x_range=[times[0], times[-1]],
                       **plot_options)
        else:
            p = figure(title=component, x_range=core_axis.x_range, 
                       **plot_options)
        p.background_fill_color = "lightgrey"
        p.yaxis.axis_label = f"{component} (mm)"
        p.xaxis.axis_label = "Time stamp (UTC)"
        p.min_border_bottom = 0
        p.min_border_top = 0
        p_scatter = p.scatter(
            x="times", y="obs", source=source, line_color=None, 
            color=data_color)
        p_band = Band(
            base="times", lower="lower", upper="upper", source=source,
            level="underlay", fill_alpha=1.0, line_width=1, 
            line_color=data_color)
        p.add_layout(p_band)

        datetick_formatter = DatetimeTickFormatter(
            days=["%Y/%m/%d"], months=["%Y/%m/%d"],
            hours=["%Y/%m/%dT%H:%M:%S"], minutes=["%Y/%m/%dT%H:%M:%S"],
            seconds=["%H:%M:%S"], hourmin=["%Y/%m/%dT%H:%M:%S"],
            minsec=["%H:%M:%S"])
        p.xaxis.formatter = datetick_formatter

        # Add picker
        picks = ColumnDataSource({"x": [], "y": [], 'color': []})
        renderer = p.scatter(
                x='x', y='y', source=picks, color='color', size=10)
        renderer_up = p.ray(
            x='x', y='y', source=picks, color='color', angle=90, 
            angle_units="deg", length=0)
        renderer_down = p.ray(
            x='x', y='y', source=picks, color='color', angle=270, 
            angle_units="deg", length=0)
        draw_tool = PointDrawTool(
            renderers=[renderer, renderer_up, renderer_down], 
            empty_value="red", num_objects=2)
        p.add_tools(draw_tool)
        p.toolbar.active_tap = draw_tool

        tabledatetimeformatter = DateFormatter(format="%Y/%m/%d")
        columns = [TableColumn(field="x", title="Date", 
                               formatter=tabledatetimeformatter),
                   TableColumn(field="y", title="Offset (mm)", 
                               formatter=NumberFormatter(format="0.00"))]
        table = DataTable(source=picks, columns=columns, editable=True, 
                          width=300)
        return p, table

    def _mpl_plot(self, output = None, *args, **kwargs):
        """ Make a matplotlib plot. """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=len(self.components), ncols=1, sharex=True)
        if len(self.components) == 1:
            axes = [axes]
        linestyle = kwargs.pop("linestyle", "None")
        marker = kwargs.pop("marker", "None")
        for channel, ax in zip(self.components, axes):
            ax.errorbar(
                self.__dict__[channel].times, 
                self.__dict__[channel].observations,
                yerr=self.__dict__[channel].errors,
                marker=marker, linestyle=linestyle, *args, **kwargs)
            ax.set(xlabel=None, ylabel="Offset (mm)", title=channel)
        axes[-1].set_xlabel("UTC Date Time")
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
        fig.suptitle = self.reciever
        if not output:
            plt.show()
        else:
            fig.savefig(output)


if __name__ == "__main__":
    import argparse

    dt_format = "%Y-%m-%d"

    parser = argparse.ArgumentParser(description="GPS Viewer applet")

    parser.add_argument(
        "-r", "--reciever", help="GPS site to get data for",
        type=str, default="LYTT")
    parser.add_argument(
        "-s", "--starttime", type=str, default="2000-01-01", 
        help=f"Start date to get data for as YYYY-MM-DD.")
    parser.add_argument(
        "-e", "--endtime", type=str, default="2020-01-01", 
        help=f"End date for when to get data for as YYYY-MM-DD.")
    parser.add_argument(
        "-o", "--output", type=str, required=False, 
        help="File output to, otherwise, will show")

    args = parser.parse_args()
    args.starttime = dt.strptime(args.starttime, dt_format)
    args.endtime = dt.strptime(args.endtime, dt_format)

    Viewer = GPSViewer(
        reciever=args.reciever, starttime=args.starttime, endtime=args.endtime)
    Viewer.plot(output=args.output)