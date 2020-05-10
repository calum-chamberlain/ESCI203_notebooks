# Make html plots for ESCI203 labs

SOURCEDIR	= helpers
BUILDDIR	= static
TARGETS		:= lab1 lab2

help:
		@echo "Use \`make <target>` where <target> is one of"
		@echo "	lab1	to make the files for lab 1"
		@echo "	lab2	to make the files for lab 2"
		@echo " all		to make all the labs"

clean:
		rm -rf $(BUILDDIR)/*

lab1:
		python $(SOURCEDIR)/bokeh_map.py -o $(BUILDDIR)/circle_location.html
		python $(SOURCEDIR)/bokeh_picker.py -o $(BUILDDIR)/picker.html

lab2:
		python $(SOURCEDIR)/gps_viewer.py -r GISB -o $(BUILDDIR)/GISB_GPS.html
		python $(SOURCEDIR)/gps_viewer.py -r HOKI -o $(BUILDDIR)/HOKI_GPS.html
		python $(SOURCEDIR)/gps_viewer.py -r LYTT -o $(BUILDDIR)/LYTT_GPS.html
		python $(SOURCEDIR)/gps_viewer.py -r TAUP -o $(BUILDDIR)/TAUP_GPS.html
		python $(SOURCEDIR)/gps_viewer.py -r TGTK -o $(BUILDDIR)/TGTK_GPS.html

all: $(TARGETS)