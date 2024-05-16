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

class SHIP(CanvasPositionMarker):
    def __init__(self, name, deg_x: float, deg_y: float,text: str = None, parent=None):
        
        if parent is None:
            super().__init__(self,(deg_x,deg_y),text=text,**kwargs)
            return #stop doing anything


        #store parent
        self.parent=parent
        self.map_widget=parent.map_widget

        #store bag name
        self.name=name
        
        #restore context
        super().__init__(self.map_widget,(deg_x,deg_y), text=text, image_zoom_visibility=(0, float("inf")), icon=self.parent.assets.icon_trash_map)
        
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

class TRASH(CanvasPositionMarker):
    def __init__(self, name, deg_x: float, deg_y: float,text: str = None, parent=None):
        
        if parent is None:
            super().__init__(self,(deg_x,deg_y),text=text,**kwargs)
            return #stop doing anything


        #store parent
        self.parent=parent
        self.map_widget=parent.map_widget

        #store bag name
        self.name=name
        
        #restore context
        super().__init__(self.map_widget,(deg_x,deg_y), text=text, image_zoom_visibility=(0, float("inf")), icon=self.parent.assets.icon_lvflower)
        
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

class TRASHTAB(tkinter.ttk.PanedWindow):
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
        self.map_widget = TkinterMapView(corner_radius=0, height=int(self.parent.screenheight*0.8)+1) #this widget has no automatic size
        self.map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}&s=Ga", max_zoom=22)
        self.GPS_panel.add(self.map_widget)
        # self.map_widget.pack_propagate(False)
        self.topleft=[37.420088,-6.001346]
        self.bottomright=[37.418399,-5.997504]
        self.map_widget.fit_bounding_box(self.topleft, self.bottomright)
        ##create info bar
        gps_data= tkinter.ttk.PanedWindow(orient="horizontal",height=1)
        self.GPS_panel.add(gps_data)
        #scale
        w = tkinter.Scale(gps_data, from_=0, to=0, orient="horizontal", length=self.parent.screenwidth*0.9,command=self.go_to_time)
        gps_data.add(w)
        #play and center buttons
        aux=tkinter.Frame(gps_data, height=1, borderwidth=1) 
        buttonsframe=tkinter.Frame(aux, height=1, borderwidth=1) #for date and play/center
        buttonsframe.pack(side="top")
        self.playbuttom = tkinter.Button(buttonsframe, image=self.assets.icon_play, command=self.play, width=35, height=35)
        self.playbuttom.pack(side="left")
        self.play_status=False
        self.playbuttom["state"] = "disabled"
        self.center_buttom = tkinter.Button(buttonsframe, text="Center", command=self.center_map, width=100, height=1)
        self.center_buttom.pack(side="right",expand="true",fill="both")



        #date
        self.date_var=tkinter.StringVar()
        self.date=tkinter.Label(aux, textvariable=self.date_var, width=int(self.parent.screenwidth*0.08), height=1, font=tkinter.font.Font(weight='bold', size=12))
        self.date.pack(side="bottom", padx=0, pady=2,anchor="c")
        
        gps_data.add(aux)

    def play(self):
        if self.play_status:
            self.playbuttom.config(image=self.assets.icon_pause)
            self.play_status=False
        else:
            self.playbuttom.config(image=self.assets.icon_play)
            self.play_status=True


    def go_to_time(self, value):
        print(value)


    def execute_time(self):
        pass


    def center_map(self):
        self.map_widget.fit_bounding_box(self.topleft, self.bottomright)

    def update_database(self):
        self.date_var.set(self.shared.rawdatabase.at[0,"Datetime"])


