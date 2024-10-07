Video Activity Tracker  
This tracker compares videos frame by frame and deletes non-moving background.   
It highlights moving objects as white. The amount of white values are calculated in
percent.

First you have to identify your regions of interest (ROIs) by clicking with the
mouse on the top left and lower right corner of the ROIs. In the end press Esc
to exit the ROI marking.

Then the video will be analysed and a precentage will be calculated. If the noise
in the data is too high play around with the starting variables BRIGHTEN, BLUR, CLAHE=False
THRESHINTERACTIVE, THRESH2, BACKGROUNDSUB.


Keys
----
In ROI marking:
d - delet elast point
ESC - finish ROI marking
In Video analysis mode:
a - play/pause playback of video
v - show original video on/off
s - show/not show (speeds up processing time)
t - threshold on/off
ESC - exit
