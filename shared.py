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