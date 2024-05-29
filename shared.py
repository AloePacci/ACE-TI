import tkinter
import tkinter.font
import tkinter.ttk
import os, sys, webbrowser, platform
import time
import traceback
import re


class SHARED():
    def __init__(self):
        self.rawdatabase=None
        self.database=None
        self.Turbidity=tkinter.DoubleVar()
        self.PH=tkinter.DoubleVar()
        self.Battery=tkinter.DoubleVar()
        self.Temperature=tkinter.DoubleVar()
        self.Conductivity=tkinter.DoubleVar()
        self.Sonar=tkinter.DoubleVar()
        self.Date=tkinter.StringVar()
        self.Date.set("                  ")
