import tkinter
import tkinter.font
import tkinter.ttk
import os, sys, webbrowser, platform
import signal
import random
import threading
import time
import screeninfo
import traceback
import re
from PIL import Image, ImageTk
from tksheet import Sheet, num2alpha as n2a
sys.path.append('../')
from shared import *
from assets import Assets
from widgets.ListboxEditable import ListboxEditable
from widgets.labels import *
import pandas as pd
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

class Table(tkinter.ttk.PanedWindow):
    def __init__(self, dataframe=None, parent=None):
        #store context
        if parent is None:
            self.parent=tkinter.Tk()
            dataframe=pd.DataFrame()
        else:
            self.parent=parent

        super().__init__(orient="horizontal")
        if dataframe==None:
            dataframe=pd.read_csv(self.parent.assets.resource_path("assets/empty.csv"),sep=",")
        self.dataframe=dataframe

        #extract data from dataframe
        headers=[]
        for i in self.dataframe:
            headers.append(i)

        self.dft= self.dataframe.T

        data=[]
        for i in range(len(self.dataframe)-1):
            data.append(self.dataframe.iloc[i])

        #create frame
        self.frame = tkinter.Frame(self.parent)
        self.frame.grid_columnconfigure(0, weight = 1)
        self.frame.grid_rowconfigure(0, weight = 1)
        self.to_hide=[]
        self.table=data
        self.headers=headers
        #create sheet
        self.sheet = Sheet(self.frame, data=self.table, headers=self.headers, width=int(self.parent.screenwidth*0.99), height=int(self.parent.screenheight*0.9))
        self.sheet.enable_bindings("single_select","drag_select","row_select","column_select","row_height_resize", "column_width_resize","arrowkeys","right_click_popup_menu","ctrl_click_select", "copy")
        
        self.frame.grid(row = 0, column = 0, sticky = "nswe") #aligned
        self.sheet.grid(row = 0, column = 0, sticky = "nswe")


        self.sheet.set_all_cell_sizes_to_text()

    def update_database(self):
        self.dataframe=self.parent.shared.database
        headers=[]
        for i in self.dataframe:
            headers.append(i)

        self.dft= self.dataframe.T

        data=[]
        for i in range(len(self.dataframe)-1):
            data.append(self.dataframe.iloc[i])

        self.table=data
        self.sheet.set_sheet_data(data=data)
        self.sheet.set_index_data(self.dataframe.index)

        self.number_of_columns=len(headers)-1

        self.sheet.set_all_column_widths(self.parent.screenwidth//(self.number_of_columns+2))
        

    def reset_Cell(self):
        current_selection = self.sheet.get_currently_selected()
        if current_selection:
            value=self.db.get_data()[current_selection.row][current_selection.column]
            box = (current_selection.row, current_selection.column)
            self.sheet[box].options(undo=True).data = value  # set cell data, end user Undo enabled
            self.sheet[box].dehighlight()

    def validate_edits(self, event):
        if not event.value:
            self.sheet_modified(list(event.cells.table.keys())[0])
            return ""
        if list(event.cells.table.values())[0] != self.sheet[list(event.cells.table.keys())[0]].data:
            self.sheet_modified(list(event.cells.table.keys())[0])
            return event.value

    def test(self):
        current_selection = self.sheet.get_currently_selected()
        print(self.db.table[current_selection.row][current_selection.column])
        # print(self.sheet[n2a(self.sheet.data_c(current_selection.column))].options(table=False, header=True).data)
        # print(self.sheet.get_column_widths(True))

    def dummy(self):
        pass

    def sheet_modified(self, box):
        self.unsaved_cells.append(box)
        self.sheet[box].bg = "indianred1"

    def compare_and_highlight(self):
        original = self.db.get_data()
        for i in range(min(len(original),len(self.db.table))):
            for j in range(min(len(original[0]),len(self.db.table[0]))):
                if original[i][j]!=self.db.table[i][j]:
                    self.sheet[(i,j)].bg = "salmon1"
                    self.unsaved_cells.append((i,j))
                else:
                    self.sheet[(i,j)].dehighlight()

    def sort_up(self):
        current_selection = self.sheet.get_currently_selected()
        if self.sheet[n2a(self.sheet.data_c(current_selection.column))].options(table=False, header=True).data in "Value, Position, Inventory, Price, ":
            self.db.table=sorted(self.db.table, key=functools.partial(self.number_from_metric,current_selection.column), reverse=True)
        else:
            self.db.table=sorted(self.db.table, key=lambda row: row[current_selection.column], reverse=True)
        self.sheet.set_sheet_data(data=self.db.table)
        self.fit_to_screen()
        self.compare_and_highlight()

    
    def sort_down(self):
        current_selection = self.sheet.get_currently_selected()
        if self.sheet[n2a(self.sheet.data_c(current_selection.column-1))].options(table=False, header=True).data in "Distance (m), Drone Lat, Drone Lon, Drone Heading, Object Lat, Object Lon, Object Heading":
            self.db.table=sorted(self.db.table, key=functools.partial(self.number_from_metric,current_selection.column))
        else:
            self.db.table=sorted(self.db.table, key=lambda row: row[current_selection.column])
        self.sheet.set_sheet_data(data=self.db.table)
        self.fit_to_screen()
        self.compare_and_highlight()

    @staticmethod
    def number_from_metric(number,row):
        base=""
        row=row[number]
        for i in row:
            if i in ".0123456789":
                base+=i
        multiplier=1
        if "k" in row:
            multiplier=1000
        elif "M" in row:
            multiplier=1000000
        elif "G" in row:
            multiplier=1000000000
        elif "m" in row:
            multiplier=0.001
        elif "u" in row:
            multiplier=0.000001
        elif "n" in row:
            multiplier=0.000000001
        elif "p" in row:
            multiplier=0.000000000001
        return float(base)*multiplier
        

    def reset(self):
        # overwrites sheet data, more information at:
        # https://github.com/ragardner/tksheet/wiki/Version-7#setting-sheet-data
        self.sheet.set_sheet_data(data=self.db.get_data())
        self.db.table=self.db.get_data()
        for box in self.unsaved_cells:
            self.sheet[box].dehighlight()
        self.unsaved_cells=[]
        self.fit_to_screen()
        

    def fit_to_screen(self):
        #search for max size of each column
        if len(self.colum_best_size) ==0: #if we have yet to calculate best size
            for i in range(1,self.number_of_columns+1): #for each column
                width=0
                #search for max size of column
                for j in range(1,len(self.table)+1):
                    print(j)
                    self.sheet.set_cell_size_to_text(j,i)
                    cols=self.sheet.get_column_widths(True)
                    if cols[i]>width:
                        width=cols[i]-cols[i-1]

                self.colum_best_size.append(width)

            for i in range(len(self.colum_best_size)):
                if self.colum_best_size[i]<140:
                    self.colum_best_size[i]=140


        #set columns to width
        for i in range(self.number_of_columns):
            self.sheet.column_width(i,self.colum_best_size[self.sheet.data_c(i)])
        sizes=self.sheet.get_column_widths(True)
        # print(f"{self.name} size {sizes}")

class TRASHLOGTAB(tkinter.ttk.PanedWindow):
    def __init__(self, parent=None):
        if parent is None:
            self.parent=tkinter.Tk()
            self.assets=Assets()
            self.shared=SHARED()
        else:
            self.parent=parent
            self.assets=self.parent.assets
            self.shared=self.parent.shared

        super().__init__(orient="horizontal")


        self.screenheight=self.parent.screenheight
        self.screenwidth=self.parent.screenwidth
        self.screenyoffset=self.parent.screenyoffset
        self.screenxoffset=self.parent.screenxoffset

        ####################################################
        ############## table
        ####################################################

        self.table=Table(dataframe=self.shared.database, parent=self)

