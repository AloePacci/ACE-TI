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
        self.map_widget = TkinterMapView(corner_radius=0, height=int(self.parent.screenheight*0.7)+1) #this widget has no automatic size
        self.map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}&s=Ga", max_zoom=22)
        self.GPS_panel.add(self.map_widget)
        # self.map_widget.pack_propagate(False)
        topleft=[37.420088,-6.001346]
        bottomright=[37.418399,-5.997504]
        self.map_widget._fit_bounding_box(topleft, bottomright)





