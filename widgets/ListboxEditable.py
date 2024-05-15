# Author: David Duran Perez
# Date: May 26, 2017

# Necessary imports
from tkinter import *
from tkinter import ttk

# Colors
colorActiveTab="#CCCCCC" # Color of the active tab
colorNoActiveTab="#EBEBEB" # Color of the no active tab
# Fonts
fontLabels='Calibri'
sizeLabels2=13

class ListboxEditable(object):
    """A class that emulates a listbox, but you can also edit a field"""
    # Constructor
    def __init__(self,frameMaster,lista):
        # *** Assign the first variables ***
        # The frame that contains the ListboxEditable
        self.frameMaster=frameMaster
        # List of the initial items
        self.list=lista
        # Number of initial rows at the moment
        self.numberRows=len(self.list)

        self.placed=False
        self.selected=1
        # *** Create the necessary labels ***
        ind=1
        for row in self.list:
            # Get the name of the label
            labelName='label'+str(ind)
            # Create the variable
            setattr(self, labelName, Label(self.frameMaster, text=self.list[ind-1], bg=colorActiveTab, fg='black', font=(fontLabels, sizeLabels2), pady=2, padx=2, width=20))

            # ** Bind actions
            # 1 left click - Change background
            getattr(self, labelName).bind('<Button-1>',lambda event, a=labelName, ind=ind: self.changeBackground(a, ind))
            # Double click - Convert to entry
            getattr(self, labelName).bind('<Double-1>',lambda event, a=ind: self.changeToEntry(a))
            # Move up and down
            getattr(self, labelName).bind("<Up>",lambda event, a=ind: self.up(a))
            getattr(self, labelName).bind("<Down>",lambda event, a=ind: self.down(a))

            # Increase the iterator
            ind=ind+1

    # Place
    def placeListBoxEditable(self):
        ind=1
        self.placed=True
        # Go row by row placing it
        for row in self.list:
            # Get the name of the label
            labelName='label'+str(ind)
            # Place the variable
            getattr(self, labelName).grid(row=ind-1,column=0)

            # Increase the iterator
            ind=ind+1

    def UpdateListBoxEditable(self,lista):
        ind=1
        self.list=lista
        for row in self.list:
            # Get the name of the label
            labelName='label'+str(ind)
            if self.placed:
                try:
                    getattr(self, labelName).grid_remove()
                except:
                    pass
            # Create the variable
            setattr(self, labelName, Label(self.frameMaster, text=self.list[ind-1], bg=colorActiveTab, fg='black', font=(fontLabels, sizeLabels2), pady=2, padx=2, width=20))

            # ** Bind actions
            # 1 left click - Change background
            getattr(self, labelName).bind('<Button-1>',lambda event, a=labelName: self.changeBackground(a))
            # Double click - Convert to entry
            getattr(self, labelName).bind('<Double-1>',lambda event, a=ind: self.changeToEntry(a))
            # Move up and down
            getattr(self, labelName).bind("<Up>",lambda event, a=ind: self.up(a))
            getattr(self, labelName).bind("<Down>",lambda event, a=ind: self.down(a))

            # Increase the iterator
            ind=ind+1

        self.placeListBoxEditable()
        self.changeBackground('label'+str(self.selected))



    # Action to do when one click
    def changeBackground(self,labelNameSelected):
        # Ensure that all the remaining labels are deselected
        ind=1
        for row in self.list:
            # Get the name of the label
            labelName='label'+str(ind)
            # Place the variable
            try:
                getattr(self, labelName).configure(bg=colorActiveTab)
            except:
                pass
            # Increase the iterator
            ind=ind+1
        # Change the background of the corresponding label
        self.selected=int(labelNameSelected[5:])
        getattr(self, labelNameSelected).configure(bg=colorNoActiveTab)
        # Set the focus for future bindings (moves)
        getattr(self, labelNameSelected).focus_set()


    # Function to do when up button pressed
    def up(self, ind):
        if ind==1: # Go to the last
            # Get the name of the label
            labelName='label'+str(len(self.list))
            self.selected=self.numberRows
        else: # Normal
            # Get the name of the label
            labelName='label'+str(ind-1)
            self.selected=ind-1

        # Call the select
        self.changeBackground(labelName)


    # Function to do when down button pressed
    def down(self, ind):
        if ind==len(self.list): # Go to the last
            # Get the name of the label
            labelName='label1'
            self.selected=1
        else: # Normal
            # Get the name of the label
            labelName='label'+str(ind+1)
            self.selected=ind+1

        # Call the select
        self.changeBackground(labelName)


    # Action to do when double-click
    def changeToEntry(self,ind):
        # Variable of the current entry
        self.entryVar=StringVar()
        # Create the entry
        #entryName='entry'+str(ind) # Name
        self.entryActive=ttk.Entry(self.frameMaster, font=(fontLabels, sizeLabels2), textvariable=self.entryVar, width=200)
        # Place it on the correct grid position
        self.entryActive.grid(sticky="we",row=ind-1,column=0)
        # Focus to the entry
        self.entryActive.focus_set()

        # Bind the action of focusOut
        self.entryActive.bind("<FocusOut>",lambda event, a=ind: self.saveEntryValue(a))

    
    # Action to do when focus out from the entry
    def saveEntryValue(self,ind):
        # Find the label to recover
        labelName='label'+str(ind)
        # Remove the entry from the screen
        self.entryActive.grid_forget()
        # Place it again
        getattr(self, labelName).grid(row=ind-1,column=0)
        # Change the name to the value of the entry
        getattr(self, labelName).configure(text=self.entryVar.get())


    def get_selected(self):
        return self.selected-1