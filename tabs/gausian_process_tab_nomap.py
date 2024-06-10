import tkinter
import tkinter.font
import tkinter.ttk
import os, sys, webbrowser, platform
import signal
import random
from tkintermapview.canvas_position_marker import CanvasPositionMarker
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import threading
import time
import screeninfo
import traceback
import re
import math
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image, ImageTk
sys.path.append('../')
from shared import *
from assets import *
from widgets.ListboxEditable import ListboxEditable
from widgets.labels import *
from submodules.database import Database
from tkcalendar import Calendar

class Checkbox(tkinter.ttk.Checkbutton):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variable = tkinter.BooleanVar(self)
        self.configure(variable=self.variable)
    
    def checked(self):
        return self.variable.get()
    
    def check(self):
        self.variable.set(True)
    
    def uncheck(self):
        self.variable.set(False)

    


class GAUSIANSENSORTAB(tkinter.ttk.PanedWindow):
    def __init__(self, parent=None):
        if parent is None:
            parent=tkinter.Tk()
        self.parent=parent
        self.busy=False
        self.checkboxes=[]
        #inherit
        try:
            self.assets=self.parent.assets
            self.shared=self.parent.shared
        except:
            self.assets=Assets()
            self.shared=SHARED()

        super().__init__(orient="horizontal")
        self.database= Database(self)
        self.taget_sensor=self.database.sensors[0]
        self.gp_step=0
        self.gs_tab()

    def gs_tab(self):
        self.GPS_panel = tkinter.ttk.PanedWindow(orient="vertical", width=int(self.parent.screenwidth*0.8))
        self.add(self.GPS_panel)

        #create map
        self.fig, self.axis = plt.subplots()
        self.map_widget = FigureCanvasTkAgg(self.fig, master=self.GPS_panel)  # A tk.DrawingArea.
        self.GPS_panel.add(self.map_widget.get_tk_widget())
        # self.map_widget.pack_propagate(False)
        self.topleft=[37.420088,-6.001346]
        self.bottomright=[37.418399,-5.997504]

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
        self.do_buttom = tkinter.Button(buttonsframe, text="do", command=self.gp_loop, width=20, height=1,font=tkinter.font.Font(weight='bold', size=18))
        self.do_buttom.pack(side="left",expand="false",fill="y")
        buttonsframe.pack(side="right",expand="false",fill="y")


        #map selection
        self.map_selection=tkinter.Frame(self, width=20)
        self.map_selection.pack_propagate(False)
        self.add(self.map_selection)
        for i in range(len(self.database.sensors)): #for as many sensors
            aux= Checkbox(self.map_selection, text=self.database.sensors[i], width=20, command=lambda a=i: self.draw_map(a))
            aux.pack(side="top")
            self.checkboxes.append(aux)
        self.checkboxes[0].check()

        self.do_playback=True
        self.playback_frame=tkinter.Frame(self.map_selection)
        self.playback_button = tkinter.Button(self.playback_frame, image = self.assets.icon_on, bd = 0, command = self.playback_data)
        self.playback_label=tkinter.Label(self.playback_frame, text="Now")
        self.playback_label.pack(side="left", anchor="e")
        self.playback_button.pack(side="left", pady = 50, anchor="w")
        self.playback_frame.pack(side="top", fill="x")

        #calendar
        self.playback_selector_frame=tkinter.Frame(self.map_selection)
        #day=datetime.now()
        day = datetime.strptime("2024-02-27", '%Y-%m-%d')
        cal = Calendar(self.playback_selector_frame, selectmode = 'day',
               year = day.year, month = day.month,
               day = day.day)
 
        cal.pack(pady = 20, expand="false",fill="none")
        self.playback_selector_frame.pack()

    def playback_data(self, event=None):
        self.do_playback = not self.do_playback
        #TODO: change to data from specific date or now
        if self.do_playback:
            self.playback_selector_frame.pack()
            self.playback_button.config(image=self.assets.icon_on)
        else:
            self.playback_button.config(image=self.assets.icon_off)
            self.playback_selector_frame.pack_forget()

    def gp_loop(self, event=None):
        if self.gp_step==0 and not self.busy: #query new data
            print("Execute 0")
            self.database.query()
            self.busy=True
        elif self.gp_step==1 and not self.busy: #get gp points
            print("Execute 1")
            self.database.generate_gp(sensor=self.taget_sensor)
            self.busy=True
        elif self.gp_step == 2:
            print("Execute 2")

            self.map_widget.draw()
            self.gp_step=0
            self.gp_loop_object=self.parent.after(60*1000, self.gp_loop) #wait a minute before generating new map
            return
        
        if self.database.busy:
            self.gp_loop_object=self.parent.after(100, self.gp_loop)
        else:
            self.busy=False
            self.gp_step+=1
            self.gp_loop_object=self.parent.after(10, self.gp_loop)

        
        
    def draw_map(self, event=None):
        for i in self.checkboxes:
            i.uncheck()
        self.checkboxes[event].check()
        self.taget_sensor=self.database.sensors[event]
        self.database.create_map(sensor=self.taget_sensor)
        self.map_widget.draw()