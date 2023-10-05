from tkinter import *
from tkinter import ttk
import os
from pathlib import Path
from PIL import Image, ImageTk
import pyautogui
import matplotlib.pyplot as plt

screen_width, screen_height = pyautogui.size()
width, height = 960, 540

folder_one = Path("Test")
first_list = list(folder_one.glob("*.jpg"))

folder_two = Path("Test - Copy")
second_list = list(folder_two.glob("*.jpg"))

counter_first = 0
counter_second = 0
def Left_Window():
    global counter_first
    counter_first += 1
    if counter_first > len(first_list)-1: counter_first = 0
    img = Image.open(first_list[counter_first])
    img = img.resize((width, height))
    stgImg = ImageTk.PhotoImage(img)
    label1.configure(image=stgImg)
    label1.image = stgImg

def Right_Window():
    global counter_second
    counter_second += 1
    if counter_second > len(second_list)-1: counter_second = 0
    img2 = Image.open(second_list[counter_second])
    img2 = img2.resize((width, height))
    stgImg2 = ImageTk.PhotoImage(img2)
    label2.configure(image=stgImg2)
    label2.image = stgImg2

def Both_Left():
    Left_Window()
    Right_Window()

def Both_Right():
    global counter_first, counter_second
    counter_first -= 1
    if counter_first < 0: counter_first = len(first_list)-1
    img = Image.open(first_list[counter_first])
    img = img.resize((width, height))
    stgImg = ImageTk.PhotoImage(img)
    label1.configure(image=stgImg)
    label1.image = stgImg

    counter_second -= 1
    if counter_second < 0: counter_second = len(second_list)-1
    img2 = Image.open(second_list[counter_second])
    img2 = img2.resize((width, height))
    stgImg2 = ImageTk.PhotoImage(img2)
    label2.configure(image=stgImg2)
    label2.image = stgImg2

def key_released(e):
    if e.keycode == 37:
        Both_Left()
    if e.keycode == 39:
        Both_Right()

root = Tk()
root.configure(bg='black',bd=0, highlightthickness=0)
frame = Frame(root, bg='black',highlightthickness=0)
frame.configure(bd=0, highlightthickness=0)
root.resizable(True, False)

color_checker_left = (0, 0, 0)
color_checker_right = (0, 0, 0)
def check_color():
    global color_checker_left
    global color_checker_right
    x, y = pyautogui.position()
    if x < (screen_width/2) - 4:
        color_checker_left = pyautogui.pixel(x, y)
        left_label = ttk.Label(text=str(color_checker_left))
        left_label.grid(column=0, row=1,sticky="news")
        left_label.configure()
        color_checker_right = pyautogui.pixel(int(((screen_width/2) + x)) + 4, y)
        right_label = ttk.Label(text=str(color_checker_right))
        right_label.grid(column=1, row=1,sticky="news")
        right_label.configure()
    elif x > (screen_width/2) + 4:
        color_checker_left = pyautogui.pixel(int((x - (screen_width/2))) - 4 , y)
        left_label = ttk.Label(text=str(color_checker_left))
        left_label.grid(column=0, row=1,sticky="news")
        left_label.configure()
        color_checker_right = pyautogui.pixel(x, y)
        right_label = ttk.Label(text=str(color_checker_right))
        right_label.grid(column=1, row=1,sticky="news")
        right_label.configure()
    root.after(100, check_color)


check_color()
# root.geometry('1010x740+200+200')

img = Image.open(first_list[counter_first])
img = img.resize((width, height))
stgImg = ImageTk.PhotoImage(img)
label1 = ttk.Label(root, image=stgImg)
label1.grid(column=0, row=0,sticky="news")


img2 = Image.open(second_list[counter_second])
img2 = img2.resize((width, height))
stgImg2 = ImageTk.PhotoImage(img2)
label2 = ttk.Label(root, image=stgImg2)
label2.grid(column=1, row=0,sticky="news")

left_label = ttk.Label(text = str(color_checker_left))
left_label.grid(column=0, row=1,sticky="news")

right_label = ttk.Label(text = str(color_checker_right))
right_label.grid(column=2, row=1,sticky="news")

left_btn = ttk.Button(root, text="Left", command=Left_Window)
left_btn.grid(column=0, row=3,sticky="news")
right_btn = ttk.Button(root, text="Right", command=Right_Window)
right_btn.grid(column=1, row=3,sticky="news")

# left_btn = ttk.Button(root, text="Left", command=Both_Left)
# left_btn.grid(column=0, row=3, sticky="news")
# right_btn = ttk.Button(root, text="Right", command=Both_Right)
# right_btn.grid(column=1, row=3, sticky="news")


root.bind('<KeyRelease>',key_released )

root.mainloop()

