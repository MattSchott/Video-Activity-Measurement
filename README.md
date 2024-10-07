# Video Activity Tracker  
![tracks](https://github.com/user-attachments/assets/46f2e7f3-ba65-4297-94ba-51e361eb4674)

This tracker compares videos frame-by-frame and removes non-moving background.
It highlights moving objects as white. The amount of white is calculated as a percentage.

First you need to identify your regions of interest (ROIs) by clicking on the upper left and lower right corner of each ROI. When finished, press Esc to exit the ROI selection.

The video will then be analysed and a percentage calculated. If the noise in the data is too high, play around with the start variables BRIGHTEN, BLUR, CLAHE, THRESHINTERACTIVE, THRESH2, BACKGROUNDSUB.

Keys  
----  
In ROI marking:  
d - delete last point  
ESC - finish ROI marking  
  
In Video analysis mode:
a - play/pause playback of video  
v - show original video on/off  
s - show/not show (speeds up processing time)  
t - threshold on/off  
ESC - exit  

### Installation  
Install python
https://www.python.org/downloads/

Install the packages opencv and numpy
Open the comand line (e.g win-key and then type cmd)
type *python -m pip install OpenCV-Python numpy*

Download all files from the github respository in the same folder

### Start the script 
by double clicking on the script or typing *python DaphniaVideoActivity_4_3.py* in the command line.
- select the example video
- Select the Petri dishes by clicking on the top left and bottom left corner of each dish.
- Press Esc to exit the ROI selection -> this will create a "Calpoints.txt" file with the coordinates. With this file in the folder, the ROIs do not need to be defined in further analysis. If you want to redefine the ROIs, delete the Calpoints.txt file.
- The analysis runs and generates a png with a screenshot of the ROIs, a screenshot of all tracks, a *-addup.txt and a *-100frame.txt.
- In the addup file, every movement leaves a white path and stays white. If the animal moves over the same area, this is not recorded. This is for recognising if animals move on a position or travel far.
- In the 100frame-file the white areas are reset every 100 frames (in a 25fps video this is every 4 seconds). This is what most people use. 

