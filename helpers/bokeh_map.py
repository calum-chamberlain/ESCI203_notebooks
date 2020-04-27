#!/usr/bin/env python3
"""
Interactive map with sliders to control circle size around locations.
"""
import numpy as np

from typing import List
from collections import namedtuple

from obspy import Inventory, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth

from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import column, row
from pyproj import Proj, transform
from bokeh.models import (
    WMTSTileSource, HoverTool, ColumnDataSource, CustomJS, Slider, 
    PointDrawTool, LabelSet)
from bokeh.tile_providers import CARTODBPOSITRON, get_provider


Station = namedtuple("station", ["code", "latitude", "longitude"])


class CircleMap():
    _proj_in = Proj(init='epsg:4326')
    _map_proj = Proj(init='epsg:3857')
    _precision = 1000 # Precision of locations in meters
    _slider_step = 0.1 # Slider step size in km
    def __init__(self, inventory: Inventory):
        self.stations = {Station(sta.code, chan.latitude, chan.longitude)
                         for net in inventory for sta in net for chan in sta}

    def __repr__(self):
        return f"CircleMap({self.stations})"

    @property
    def lats(self) -> List[float]:
        return [s.latitude for s in self.stations]

    @property
    def lons(self) -> List[float]:
        return [s.longitude for s in self.stations]

    @property
    def codes(self) -> List[str]:
        return [s.code for s in self.stations]
    
    @property
    def xy(self):
        return (transform(self._proj_in, self._map_proj, 
                          s.longitude, s.latitude) for s in self.stations)
    
    @property
    def x(self) -> List[float]:
        return [x for x, y in self.xy]

    @property
    def y(self) -> List[float]:
        return [y for x, y in self.xy]

    def plot(
        self,
        plot_height: int = 900, 
        plot_width: int = 900,
        output: str = None,
    ):
        # Set up map options
        map_hover = HoverTool(
            tooltips=[
                ("Latitude", "@lats"),
                ("Longitude", "@lons")], names=["locs"])
        map_options = dict(
            tools=[map_hover, "pan,wheel_zoom,reset"],
            plot_width=plot_width, plot_height=plot_height)
        # Get plot bounds in web mercator
        try:
            min_lat, min_lon, max_lat, max_lon = (
                min(self.lats), min(self.lons), max(self.lats), max(self.lons))
        except ValueError as e:
            print(e)
            print("Setting map bounds to NZ")
            min_lat, min_lon, max_lat, max_lon = (-47., 165., -34., 179.9)
        # Add a 10% pad
        lat_pad = 0.25 * (max_lat - min_lat)
        lon_pad = 0.25 * (max_lon - min_lon)
        min_lat = min_lat - lat_pad
        max_lat = max_lat + lat_pad
        min_lon = min_lon - lon_pad
        max_lon = max_lon + lon_pad

        bottom_left = transform(
            self._proj_in, self._map_proj, min_lon, min_lat)
        top_right = transform(
            self._proj_in, self._map_proj, max_lon, max_lat)
        map_x_range = (bottom_left[0], top_right[0])
        map_y_range = (bottom_left[1], top_right[1])

        # Make map
        map_plot = figure(
            title="Seismographs - WebMercator projection bias: distances somewhat inaccurate",
            x_range=map_x_range, y_range=map_y_range,
            x_axis_type="mercator", y_axis_type="mercator", match_aspect=True,
            **map_options)
        url = 'http://a.basemaps.cartocdn.com/rastertiles/voyager/{Z}/{X}/{Y}.png'
        attribution = "Tiles by Carto, under CC BY 3.0. Data by OSM, under ODbL"
        map_plot.add_tile(WMTSTileSource(url=url, attribution=attribution))
        # tile_provider = get_provider(CARTODBPOSITRON)
        # map_plot.add_tile(tile_provider)

        # Plot stations
        station_source = ColumnDataSource({
            'y': self.y, 'x': self.x, 'lats': self.lats, 'lons': self.lons,
            'codes': self.codes})
        map_plot.triangle(
            x="x", y="y", size=10, color="blue", alpha=1.0, 
            source=station_source, name="stations")
        labels = LabelSet(
            x="x", y="y", text="codes", level="glyph", x_offset=2, y_offset=2, 
            source=station_source, render_mode="css", text_font_size="20px")
        map_plot.add_layout(labels)
        
        # Hack to make invisbile circles of possible locations
        lat_locs, lon_locs, x_locs, y_locs = [], [], [], []
        x_range = map_x_range[1] - map_x_range[0]
        y_range = map_y_range[1] - map_y_range[0]
        for x in np.arange(map_x_range[0] - x_range, map_x_range[1] + x_range, self._precision):
            for y in np.arange(map_y_range[0] - y_range, map_y_range[1] + y_range, self._precision):
                x_locs.append(x)
                y_locs.append(y)
                lon, lat = transform(self._map_proj, self._proj_in, x, y)
                lat_locs.append(lat)
                lon_locs.append(lon)
        possible_locations = ColumnDataSource({
            'y': y_locs, 'x': x_locs, 'lats': lat_locs, 'lons': lon_locs,
            'codes': ["location" for _ in x_locs]})
        map_plot.rect(
            x="x", y="y", width=self._precision, height=self._precision, 
            source=possible_locations, fill_color="pink", fill_alpha=0.0, 
            line_color="black", line_alpha=0.0, name="locs")

        # Make circles
        sliders = []
        for station, x, y in zip(self.stations, self.x, self.y):
            # Work out a rough bias in distance due to Web Mercator projection
            dist, _, _ = gps2dist_azimuth(
                station.latitude, station.longitude,
                station.latitude, station.longitude + 1)
            # dist is distance in km for 1 degree
            _x, _y = transform(
                self._proj_in, self._map_proj, station.longitude + 1,
                station.latitude)
            map_dist = abs(x - _x)
            scale_factor = map_dist / dist
            _source = ColumnDataSource({
                "x": [x], "y": [y], "lats": [station.latitude],
                "lons": [station.longitude], "codes": [station.code],
                "radius": [10000]})
            circle = map_plot.circle(
                x='x', y='y', source=_source, radius=10000 * scale_factor, 
                line_color="red", fill_alpha=0.0, line_alpha=1.0,
                radius_dimension="x")
            
            slider = Slider(
                start=1, end=100, value=10.0, step=self._slider_step, 
                title=f"{station.code} radius (km)")

            callback = CustomJS(
                args=dict(renderer=circle), 
                code=f"renderer.glyph.radius = cb_obj.value * 1000 * {scale_factor};")

            slider.js_on_change('value', callback)            
            sliders.append(slider)

        layout = row(map_plot, column(*sliders))

        if output:
            output_file(output)
            save(layout)
        else:
            show(layout)


def circle_map(
    stations: List[str] = None,
    client: Client = None,
    time: UTCDateTime = None,
    output: str = None,
):
    import warnings

    stations = stations or ["CMWZ", "BSWZ", "TCW", "BHW", "WEL"]
    client = client or Client("GEONET")
    time = time or UTCDateTime(2019, 12, 8)

    bulk = [("*", sta, "*", "*", UTCDateTime(0), UTCDateTime()) 
            for sta in stations]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # GeoNet doesn't pass xml version properly
        inv = client.get_stations_bulk(bulk, level="channel")

    # Select the correct time and just vertical channels
    inv = inv.select(time=time)
    # Take either EHZ or HHZ or SHZ or BNZ channels
    acceptable_channels = ("EHZ", "HHZ")
    for net in inv:
        for sta in net:
            sta.channels = [chan for chan in sta.channels 
                            if chan.code in acceptable_channels]


    CircleMap(inventory=inv).plot(output=output)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive map for simple circle-based earthquake "
                    "location")
    
    parser.add_argument(
        "-s", "--stations", nargs="+", 
        help="Station names to plot - station locations will be aquired from"
             " client", default=["CMWZ", "BSWZ", "TCW", "BHW", "WEL"])
    parser.add_argument(
        "-c", "--client", type=str, help="FDSN client name or URL",
        default="GEONET")
    parser.add_argument(
        "-t", "--time", type=str, 
        help="UTCDateTime parseable string for when to plot the stations for.",
        default="2019-12-08")
    parser.add_argument(
        "-o", "--output", type=str, required=False, 
        help="File output to, otherwise, will show")

    args = parser.parse_args()
    args.time = UTCDateTime(args.time)

    client = Client(args.client)
    circle_map(stations=args.stations, client=client, time=args.time, 
               output=args.output)
