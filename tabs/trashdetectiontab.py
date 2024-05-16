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
max_retain_data=40

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
        super().__init__(self.map_widget,(deg_x,deg_y), text=text, image_zoom_visibility=(0, float("inf")),  icon=ImageTk.PhotoImage(self.parent.assets.image_ship),command=self.leftclick)
        
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

    def update_rotation(self,degrees):
        self.change_icon(ImageTk.PhotoImage(self.parent.assets.image_ship.rotate(degrees)))
class TRASH(CanvasPositionMarker):
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
    
    def update_position(self, value, lat, lon):
        self.hide_image(False)
        self.last_value=value
        self.set_position(lat,lon)


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


        self.trashes=[]
        self.ships=[]

        ##create info bar
        gps_data= tkinter.ttk.PanedWindow(orient="horizontal",height=1)
        self.GPS_panel.add(gps_data)
        #scale
        self.timeline = tkinter.Scale(gps_data, from_=0, to=0, orient="horizontal", length=self.parent.screenwidth*0.9,command=self.go_to_time)
        gps_data.add(self.timeline)
        self.play_callback=None
        #play and center buttons
        aux=tkinter.Frame(gps_data, height=1, borderwidth=1) 
        buttonsframe=tkinter.Frame(aux, height=1, borderwidth=1) #for date and play/center
        buttonsframe.pack(side="top")
        self.playbuttom = tkinter.Button(buttonsframe, image=self.assets.icon_play, command=self.play, width=35, height=35)
        self.playbuttom.pack(side="left")
        self.play_status=False
        self.playbuttom["state"] = "disabled"
        self.center_buttom = tkinter.Button(buttonsframe, text="Center", command=self.center_map, width=10, height=1)
        self.center_buttom.pack(side="right",expand="true",fill="both")
        self.speed_buttom = tkinter.Label(buttonsframe, text="x1", relief="raised" , width=5, height=1)
        self.speed_buttom.bind("<Button-1>", self.increase_speed)
        self.speed_buttom.bind("<Button-3>", self.decrease_speed)
        self.speed=1
        self.speed_buttom.pack(side="right",expand="true",fill="y")


        #date
        self.date_var=tkinter.StringVar()
        self.date=tkinter.Label(aux, textvariable=self.date_var, width=int(self.parent.screenwidth*0.08), height=1, font=tkinter.font.Font(weight='bold', size=12))
        self.date.pack(side="bottom", padx=0, pady=2,anchor="c")
        
        gps_data.add(aux)

    def play(self):
        if self.play_status:
            self.playbuttom.config(image=self.assets.icon_play)
            self.parent.after_cancel(self.play_callback)
            self.play_status=False
        else:
            if self.timeline.get()-1 == self.shared.database.index[-1]:
                self.timeline.set(0)
            self.playbuttom.config(image=self.assets.icon_pause)
            self.play_callback=self.parent.after(200//self.speed,self.execute_time)
            self.play_status=True


    def go_to_time(self, value):
        value=int(value)
        if value>=self.shared.database.index[-1]:
            return
        #get next time
        self.date_var.set(self.shared.rawdatabase.at[value,"Datetime"])
        self.ships[0].update_rotation(self.shared.rawdatabase.at[value,"Drone Heading"])
        self.ships[0].set_position(self.shared.rawdatabase.at[value,"Drone Lat"], self.shared.rawdatabase.at[value,"Drone Lon"])
        if not math.isnan(self.shared.rawdatabase.at[value,"Object Lon"]):
            processed=False
            #process trash
            for i in range(len(self.trashes)):
                if self.trashes[i].name==self.shared.rawdatabase.at[value,"Class"]:
                    processed=True#update trash
                    self.trashes[i].update_position(value, self.shared.rawdatabase.at[value,"Object Lat"], self.shared.rawdatabase.at[value,"Object Lon"])
                    break
            if not processed: #append new trash
                aux=TRASH(self.shared.rawdatabase.at[value,"Class"], self.shared.rawdatabase.at[value,"Object Lat"], self.shared.rawdatabase.at[value,"Object Lon"],parent=self)
                aux.last_value=value
                self.trashes.append(aux)
        for i in self.trashes:
            i.update(value)



        


    def execute_time(self):
        if self.timeline.get()-1 == self.shared.database.index[-1] and self.play_status:
            self.play() #stop time
            return
        self.timeline.set(self.timeline.get()+1)
        self.play_callback=self.parent.after(200//self.speed,self.execute_time)
        
    def increase_speed(self, event=None):
        if self.speed<8:
            self.speed+=1
        self.speed_buttom.config(text=f"x{self.speed}")

    def decrease_speed(self, event=None):
        if self.speed>1:
            self.speed-=1
        self.speed_buttom.config(text=f"x{self.speed}")


    def center_map(self,event=None):
        self.map_widget.fit_bounding_box(self.topleft, self.bottomright)

    def update_database(self):
        self.date_var.set(self.shared.rawdatabase.at[0,"Datetime"])
        self.timeline.config(to=len(self.shared.rawdatabase))
        self.playbuttom.config(state="normal")
        self.ships.append(SHIP("vehicle_1", self.shared.rawdatabase.at[0,"Drone Lat"], self.shared.rawdatabase.at[0,"Drone Lon"],parent=self))
        self.ships[0].update_rotation(30)


