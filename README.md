# Video Activity Tracker  
![tracks](https://github.com/user-attachments/assets/46f2e7f3-ba65-4297-94ba-51e361eb4674)

This tracker compares videos frame by frame and deletes non-moving background.   
It highlights moving objects as white. The amount of white values are calculated in
percent.

First you have to identify your regions of interest (ROIs) by clicking with the
mouse on the top left and lower right corner of the ROIs. In the end press Esc
to exit the ROI marking.

Then the video will be analysed and a precentage will be calculated. If the noise
in the data is too high play around with the starting variables BRIGHTEN, BLUR, CLAHE,
THRESHINTERACTIVE, THRESH2, BACKGROUNDSUB.


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

Install opencv and numpy
Open the comand line (e.g win-key and then type cmd)
type *python -m pip install OpenCV-Python numpy*

Download all files from the respository in the same folder

### Start Skript 
by Doubleclicking on script or in the command line type *python DaphniaVideoActivity_4_3.py*

