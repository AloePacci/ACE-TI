import customtkinter
import tkinter
import tkinter.font
import tkinter.ttk
import os, sys, webbrowser, platform
import signal
import serial
import serial.tools.list_ports
import threading
import time
import screeninfo
import traceback
import re
from PIL import Image, ImageTk
from assets import Assets
from shared import SHARED
from widgets.ListboxEditable import ListboxEditable
from widgets.labels import *
import pandas as pd
from tabs.trashdetectiontab import TRASHTAB
from tabs.trashdetectionlogtab import TRASHLOGTAB
import math
# import machine this is for hard reset (machine.reset())
version = "1.0.0"

class ACETI_GUI(tkinter.Tk):
    def __init__(self):
        self.platform=platform.system()
            #get platform

        super().__init__()
        #INIT SHAPE  todo:use customTK
        # customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
        # customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green
        self.title("ACETI GUI")
        if self.platform == "Linux":
            self.attributes('-zoomed', True)
        else:
            self.state('zoomed')
            # elif platform.system() == “Windows”:
        self.configure(bg = 'beige')
        self.option_add('*tearOff', False)  # Deshabilita submenús flotantes


        #VARIABLES FOR FUNCTIONS
        self.fullscreenstate = False
        self.assets=Assets()
        self.shared=SHARED()

        #get screen size
        monitors = screeninfo.get_monitors()
        self.screenheight=monitors[0].height
        self.screenwidth=monitors[0].width
        self.screenyoffset=monitors[0].y
        self.screenxoffset=monitors[0].x
        for m in reversed(monitors):
            if m.x <= self.winfo_x() <= m.width + m.x and m.y <= self.winfo_y() <= m.height + m.y:
                self.screenheight=m.height
                self.screenwidth=m.width
                self.screenyoffset=m.y
                self.screenxoffset=m.x


        self.mainFrame = tkinter.ttk.PanedWindow(orient="vertical")
        self.mainFrame.pack(side='top', fill='both', expand=True,)
        self.iconphoto(self, self.assets.aceti_icon)

        #menubar
        self.toolbar = tkinter.Frame(self.mainFrame, relief='raised', bd=2, bg="#E5E5E5")
        self.mainFrame.add(self.toolbar)
        self.menubar_buttons=[]
        self.button_trash_detection = tkinter.Button(self.toolbar, image=self.assets.icon_water_trash, command=lambda a=0: self.change_tab(a), width=40, height=40)
        self.menubar_buttons.append(self.button_trash_detection)
        self.button_raw_trash_detection = tkinter.Button(self.toolbar, image=self.assets.icon_trash_log, command=lambda a=1: self.change_tab(a), width=40, height=40)
        self.menubar_buttons.append(self.button_raw_trash_detection)

        for i in self.menubar_buttons:
            i.pack(side='left', expand=False,)

        ####################################################
        ############## different tabs
        ####################################################
        self.tabs=[]
        
        #create trash map tab
        self.TRASHTAB = TRASHTAB(parent=self) 
        self.tabs.append(self.TRASHTAB)

        #create trash log tab
        self.TRASHLOGTAB=TRASHLOGTAB(parent=self)
        self.tabs.append(self.TRASHLOGTAB)

        #select init tab
        self.change_tab(0)

        #MENU BAR
        menubar = tkinter.Menu(self)
        self['menu']=menubar
        
        filemenu = tkinter.Menu(menubar)
        settings = tkinter.Menu(menubar)
        helpmenu = tkinter.Menu(menubar)
        menubar.add_cascade(menu=filemenu, label='File')
        menubar.add_cascade(menu=settings, label='Settings')
        menubar.add_cascade(menu=helpmenu, label='Help')

        #FILEMENU
        filemenu.add_command(label='Load', 
                            command=self.read_database, 
                            underline=0, accelerator="Ctrl+a",
                            image=self.assets.icon_trash_log, compound='left')
        filemenu.add_separator()  # Agrega un separador
        filemenu.add_command(label='Exit', command=self.destroy, 
                            underline=0, accelerator="Ctrl+q",
                            image=self.assets.icon_trash_log, compound='left')

        #KEY BINDS
        self.bind("<F11>", self.toggle_fullscreen)
        self.bind("<Escape>", self.end_program)
        self.bind("<Control-a>", self.read_database)
        
        #extravars
        boldfont = tkinter.font.Font(weight='bold')



    def toggle_fullscreen(self, event=None):
        self.fullscreenstate = not self.fullscreenstate  # Just toggling the boolean
        self.attributes("-fullscreen", self.fullscreenstate)
        return "break"

    def end_program(self, event=None):
        self.destroy()
        return "break"

    def dummy(self):
        pass


    def change_tab(self, tab):
        for i in range(len(self.menubar_buttons)):
            if i == tab:
                self.menubar_buttons[tab].config(relief="sunken")
            else:
                self.menubar_buttons[i].config(relief="raised")
        for i in range(len(self.tabs)):
            try:
                if tab==i:
                    self.mainFrame.add(self.tabs[i])
                else:
                    self.mainFrame.forget(self.tabs[i])
            except:
                # error= traceback.format_exc()
                # print(f"There was an error oculting tab\n{error}")
                pass

    def read_database(self, event=None):
        self.path=os.getcwd()
        self.path = tkinter.filedialog.askdirectory(
            parent=self,
            title="Open Database",
            initialdir=self.path,
        )
        #check that we have files inside and load them
        files = os.listdir(self.path)
        dfs = []
        for file in files:
            if file.endswith('.log'):
                df = pd.read_csv(self.path + '/' + file)
                dfs.append(df)

        if len(dfs) == 0:
            return False #return false as we were not able to load files
        
        self.shared.rawdatabase = pd.concat(dfs,ignore_index=True)
        self.shared.database=self.shared.rawdatabase[self.shared.rawdatabase['Object Lon'].notnull()]
        self.TRASHLOGTAB.table.update_database()
        self.TRASHTAB.update_database()
        return True


def main(args=None):
    web=ACETI_GUI()
    signal.signal(signal.SIGINT, exit_gracefully)
    print(f"Init Version {version}")
    web.mainloop() 
    print("Finishing Application")
    print("Application Finished")
    

def exit_gracefully(signum, frame):
    # restore the original signal handler as otherwise evil things will happen
    # in raw_input when CTRL+C is pressed, and our signal handler is not re-entrant
    signal.signal(signal.SIGINT, original_sigint)

    try:
        print("Application Finished")

    except KeyboardInterrupt:
        print("Ok ok, quitting")
        sys.exit(1)


original_sigint_handler = signal.getsignal(signal.SIGINT)
if __name__ == '__main__':
    main()
