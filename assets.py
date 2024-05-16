import tkinter
import tkinter.font
import tkinter.ttk
import os, sys, webbrowser, platform
import time
import traceback
import re
from PIL import Image, ImageTk
import PIL.ImageTk
from itertools import count
from datetime import datetime
import threading


class GIF():
    def __init__(self, file, speed=None, size = 0):
        self.path=file
        self.loc=0
        self.frames=[]
        threading.Timer(0,self.__read__,args=(speed,size)).start()

    def frame(index):
        return self.frames[index]

    def get_next_frame(self):
        self.loc=(self.loc+1)%self.len
        return self.frames[self.loc]

    def __read__(self, speed, size):
        self.image=Image.open(self.path)
        if size == 0:
            try:
                n_frames=self.image.n_frames
            except EOFError:
                error= traceback.format_exc()
                print(f"error importing asset gif {self.path}:\n {error}")
                return

        for i in range(n_frames):
            self.frames.append(ImageTk.PhotoImage(self.image.copy()))
            self.image.seek(i)
        if speed is not None:
            self.delay=1000/speed
        else:
            try:
                self.delay = self.info.duration
            except:
                self.delay = 100

        if len(self.frames)==0:
            return
        self.len=len(self.frames)
        # print(f"gif {path.split('/')[-1]} len {self.len} delay {self.delay}")



class Assets(object):
    def __init__(self, path=None):
        self.path=self.resource_path("assets/")
        
        self.aceti_icon = self.include_image("icon_aceti.png",40,40)
        self.icon_water_trash = self.include_image("icon_water_trash.png",40,40)
        self.icon_trash_log = self.include_image("icon_trash_log.png",40,40)
        self.icon_play = self.include_image("play.png",35,35)
        self.icon_pause = self.include_image("pause.png",35,35)
        self.icon_ship = self.include_image("icon_ship.png",40,40)
        self.image_ship = Image.open(self.path+"icon_ship.png").convert('RGBA').resize((60,60))
        self.waste_len=2
        self.icon_waste=[]
        for i in range(self.waste_len):
            self.icon_waste.append(self.include_image(f"waste{i}.png",40,40))

    def include_image(self,image,sizex,sizey):
        try:
            originalImg = Image.open(self.path+image)
            img= originalImg.resize((sizex, sizey))
            return ImageTk.PhotoImage(img)
        except:
            error = traceback.format_exc()
            print(f"error importing asset {image}:\n {error}")


    def resource_path(self, relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)