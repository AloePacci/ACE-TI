import tkinter
import tkinter.font
import tkinter.ttk
import os, sys, webbrowser, platform
import signal
import random
from tkintermapview import TkinterMapView
from tkintermapview.canvas_position_marker import CanvasPositionMarker
import threading
import time
import screeninfo
import traceback
import re
import math
import pandas as pd
from PIL import Image, ImageTk
sys.path.append('../')
from shared import *
from assets import *
from widgets.ListboxEditable import ListboxEditable
from widgets.labels import *

class MEASURE(CanvasPositionMarker):
    def __init__(self, name, deg_x: float, deg_y: float,text: str = None, parent=None):
        
        if parent is None:
            super().__init__(self,(deg_x,deg_y),text=text,**kwargs)
            return #stop doing anything


        #store parent
        self.parent=parent
        self.map_widget=parent.map_widget
        self.last_value=-200
        #store bag name
        self.name=name
        
        #restore context
        super().__init__(self.map_widget,(deg_x,deg_y), text=text, image_zoom_visibility=(0, float("inf")),command=self.leftclick, icon=self.parent.assets.icon_waste[random.randint(0, self.parent.assets.waste_len-1)])
        
        #draw it
        self.draw()
        self.map_widget.canvas_marker_list.append(self)

    def mouse_enter(self,event=None):
        self.set_text(self.name)
        super().mouse_enter(event)
        self.draw()

    def mouse_leave(self, event=None):
        self.set_text("")
        super().mouse_leave(event)

    def leftclick(self, event=None):
        pass

    def update(self, value):
        if abs(value-self.last_value)>max_retain_data:
            self.hide_image(True)


class MISSIONTAB(tkinter.ttk.PanedWindow):
    def __init__(self, parent=None):
        if parent is None:
            parent=tkinter.Tk()
        self.parent=parent
        self.trash_markers=[]
        #inherit
        try:
            self.assets=self.parent.assets
            self.shared=self.parent.shared
        except:
            self.assets=Assets()
            self.shared=SHARED()

        super().__init__(orient="horizontal")

        self.trash_tab()

    def trash_tab(self):
        self.GPS_panel = tkinter.ttk.PanedWindow(orient="vertical")
        self.add(self.GPS_panel)

        
        #create map
        self.map_widget = TkinterMapView(corner_radius=0, height=int(self.parent.screenheight*0.75)+1) #this widget has no automatic size
        self.map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}&s=Ga", max_zoom=22)
        self.GPS_panel.add(self.map_widget)
        # self.map_widget.pack_propagate(False)
        self.topleft=[37.420088,-6.001346]
        self.bottomright=[37.418399,-5.997504]
        self.map_widget.fit_bounding_box(self.topleft, self.bottomright)


        self.trashes=[]
        self.ships=[]

        ##create info bar
        gps_data= tkinter.ttk.PanedWindow(orient="horizontal",height=1)
        self.GPS_panel.add(gps_data)
        #last value frame
        self.sensordata = tkinter.Frame(gps_data)
        Sensors=["Turbidity", "PH", "Battery", "Temperature", "Conductivity", "Sonar"]
        aux=create_labelh_with_units(self.sensordata, "Turbidity", self.shared.Turbidity, "NTU", expand="true")
        aux=create_labelh(self.sensordata, "PH", self.shared.PH, expand="true")
        aux=create_labelh_with_units(self.sensordata, "Battery:", self.shared.Battery, "%", expand="true")
        aux=create_labelh_with_units(self.sensordata, "Temperature:", self.shared.Temperature, "ºC", expand="true")
        aux=create_labelh_with_units(self.sensordata, "Conductivity:", self.shared.Conductivity, "S/cm", expand="true")
        aux=create_labelh_with_units(self.sensordata, "Sonar:", self.shared.Sonar, "m", expand="true")
        aux=create_labelh(self.sensordata, "Date:", self.shared.Date, expand="true")
        self.sensordata.pack(side="left",expand="true",fill="both")
        
        #play and center buttons
        buttonsframe=tkinter.Frame(gps_data, height=1, borderwidth=1) #for date and play/center
        self.center_buttom = tkinter.Button(buttonsframe, text="Center", command=self.center_map, width=20, height=1,font=tkinter.font.Font(weight='bold', size=18))
        self.center_buttom.pack(side="left",expand="false",fill="y")
        buttonsframe.pack(side="right",expand="false",fill="y")


    def center_map(self,event=None):
        self.map_widget.fit_bounding_box(self.topleft, self.bottomright)


