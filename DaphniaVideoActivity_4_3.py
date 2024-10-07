#!/usr/bin/env python

'''
Activity tracker
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
if ch == 27:
a - play/pause playback of video
v - show original video on/off
s - show/not show (speeds up processing time)
t - threshold on/off
ESC - exit
'''

import numpy as np
import cv2
import video
from common import anorm2, draw_str, getsize
#from time import clock, sleep
import os
from shutil import copyfile
import csv
#from matplotlib import pyplot as plt


####################################################### ###
#### Set starting vaiables ############################ ###
####################################################### ###
Version = 4.3
image = None
actualize = False
pt = None
add_remove_pt = False
show = True
xmouse = 0
ymouse = 0
batch = False # uncommented line 342 for Batch no show
batchcalibration = True 
BRIGHTEN = False
BLUR = 5
CLAHE=False
THRESHINTERACTIVE = False
THRESH2 = False # 150 without CLAHE
Colordetection=False
BACKGROUNDSUB = True
skip = 20 # Number of frames to skip

####################################################### ###
#### Defs ############################################# ###
####################################################### ###

### Def Create Blank ##################################
def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image2 = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image2[:] = color

    return image
#######################################################

### Def on_mouse ######################################
def on_mouse (event, x, y, flags, param):

# we will use the global pt and add_remove_pt
    	global pt
    	global add_remove_pt
    	global actualize
    	if event == cv2.EVENT_LBUTTONDOWN: 
        	# user has click, so memorize it
        	pt = (x, y)
        	add_remove_pt = True
#		print("Clicked")
#		print (pt)


#######################################################
### differences #######################################
def differ (frame, old, blur = 3, thresh = 70,auto = False):
		grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		grey= cv2.blur(grey, (blur,blur))
		if auto:
			(thresh, im_bw) = cv2.threshold(grey, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		grey = cv2.threshold(grey, thresh, 255, cv2.THRESH_BINARY)[1]
		diff = cv2.absdiff(grey, old)
		return grey, diff, thresh
###CLAHE #########################
def clahe(img):
  img= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  l, a, b = cv2.split(img)
  clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
  cl = clahe.apply(l)
  limg = cv2.merge((cl,a,b))
  final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
  return final

### build_lappyr ######################################
#def build_lappyr(img, leveln=6, dtype=np.int16):
#    	img = dtype(img)
#    	levels = []
#    	for i in xrange(leveln-1):
#        	next_img = cv2.pyrDown(img)
#        	img1 = cv2.pyrUp(next_img, dstsize=getsize(img))
#        	levels.append(img-img1)
#        	img = next_img
#    	levels.append(img)
#    	return levels
#	del next_img
#	del img1
#	del img
#	del levels

#######################################################
def mousePosition(event,x,y,flags,param):
    global xmouse
    global ymouse
    global pt
    global add_remove_pt
    global actualize
    if event == cv2.EVENT_MOUSEMOVE:
        xmouse = x
        ymouse = y
    elif event == cv2.EVENT_LBUTTONDOWN:
        	# user has click, so memorize it
        pt = (x, y)
        add_remove_pt = True
#	print("Clicked")
#	print (pt)
### merge_lappyr ######################################
#def merge_lappyr(levels):
#    	img = levels[-1]
#    	for lev_img in levels[-2::-1]:
#        	img = cv2.pyrUp(img, dstsize=getsize(lev_img))
#        	img += lev_img
#    	return np.uint8(np.clip(img, 0, 255))
#######################################################

####################################################### ###
#### Defs end########################################## ###
####################################################### ###

####################################################### ###
#### App ############################################## ###
####################################################### ###

class App:
    def __init__(self, video_src):
#        self.track_len = 10
#        self.detect_interval = 5
        self.tracks = []
        self.cam = video.create_capture(video_src)
#        self.frame_idx = 0
        self.name = video_src




    def run(self):
	
	############## Get/Set Variables #############
    	global add_remove_pt
    	global actualize
    	global pt
    	global show
    	global xmouse
    	global ymouse
    	global batch
    	global skip
############################################################################################################################################################################### #####################################
    	THRESH = False #was 70 /200
############################################################################################################################################################################### #####################################
    	fc = 0
    	filename = os.path.basename(self.name)
    	windowname = 'Activity in ' + filename
    	savename = filename[:-4] + "_"+ "V4_3-addup" + ".txt"
    	savename2 = filename[:-4] + "_"+ "V4_3-100frames" + ".txt"
#    	savename3 = filename[:-4] + "_"+ "V4_1-maxvalue" + ".txt"
    	calname = filename[:-4] + "_"+ "V4_3-cal" + ".txt"
    	picname= filename[:-4] + "_"+ "V4_3-tracks" + ".png"
### Batch calibration
    	if batchcalibration:
    	  calname = "Calpoints.txt"
### Batchcalibration end
    	filepath = os.path.dirname(self.name)
    #        print filepath
    	savename = filepath + "/" + savename
    	savename2 = filepath + "/" + savename2
#    	savename3 = filepath + "/" + savename3
    	calname2 = filepath + "/" + calname
    	calname = filepath + "/" + calname
    	picname = filepath + "/" + picname
    	while os.path.exists(savename):
    		savename = savename[:-4] + "I.txt"
    		savename2 = savename2[:-4] + "I.txt"
#    		savename3 = savename3[:-4] + "I.txt"
    	image = None
    	n = 2 # set number of observations in video
    	WaitTime = 10
    	ptsli =()
    	ptsre =()
    #	if show:
    #		log = False
    #		print "No log just show"
    #	else:
    #		log = True
    	log = True
    	############## Generate fileheader ############
    	fileheader = "fc; "
    #	for i in xrange(n):
    #	 	fileheader  = fileheader  + "meangrey" + str(i) + "; "
    	###############################################
    	fcmax = int(self.cam.get(cv2.CAP_PROP_FRAME_COUNT))
    	if fcmax < 0:
            fcmax = 1500
    	if (self.cam.get(cv2.CAP_PROP_FPS)) != (self.cam.get(cv2.CAP_PROP_FPS)):
    		print((str(fcmax)) + " frames of video to analyse - fps unknown")
    	############## Save logfile with header #######
    		if log:
    #			print "DEBUG", savename3
    			text_file = open(savename, "w")
    			text_file.write("# " + filename + " with " + str(fcmax) + " frames, with unknown fps")
    			text_file3 = open(savename2, "w")
    			text_file3.write("# " + filename + " with " + str(fcmax) + " frames, with unknown fps")
    			# text_file4 = open(savename3, "w")
    			# text_file4.write("# " + filename + " with " + str(fcmax) + " frames, with unknown fps")
    	else:
    		print((str(fcmax)) + " frames - " + str(fcmax/int(self.cam.get(cv2.CAP_PROP_FPS))) + " sec of video to analyse")
    	############## Save logfile with header #######
    		if log:
    #			print "DEBUG", savename3
    			text_file = open(savename, "w")
    			text_file.write("# " + filename + " with " + str(fcmax) + " frames, " + str(int(self.cam.get(cv2.CAP_PROP_FPS)))+ " fps,"+ str(fcmax/int(self.cam.get(cv2.CAP_PROP_FPS))) + " sec" + "\n")
    			text_file3 = open(savename2, "w")
    			text_file3.write("# " + filename + " with " + str(fcmax) + " frames, " + str(int(self.cam.get(cv2.CAP_PROP_FPS)))+ " fps,"+ str(fcmax/int(self.cam.get(cv2.CAP_PROP_FPS))) + " sec" + "\n")
    			# text_file4 = open(savename3, "w")
    			# text_file4.write("# " + filename + " with " + str(fcmax) + " frames, " + str(int(self.cam.get(cv2.CAP_PROP_FPS)))+ " fps,"+ str(fcmax/int(self.cam.get(cv2.CAP_PROP_FPS))) + " sec" + "\n")
    	###############################################
    #	if show:
    #	cv2.namedWindow ("links",cv2.WINDOW_NORMAL)
    #	cv2.namedWindow ("rechts",cv2.WINDOW_NORMAL)
    #	cv2.namedWindow ("original",cv2.WINDOW_NORMAL)
    
    #	cv2.namedWindow ("Red",cv2.WINDOW_NORMAL)
    #	cv2.namedWindow ("Blue",cv2.WINDOW_NORMAL)
    #	cv2.namedWindow ("Green",cv2.WINDOW_NORMAL)
    #	cv2.namedWindow ("cropped",cv2.WINDOW_NORMAL)
    	fcmax = fcmax -30 #### to avoid floating point error ####
    	cv2.namedWindow (windowname,cv2.WINDOW_NORMAL)
    	font = cv2.FONT_HERSHEY_SIMPLEX
    	calibrate = True
    	calpoints = ();
    	letters = list(map(chr, list(range(97, 123)))) # for differentiating data of each individual 
    	for l in range(len(letters)):
    		letters = letters + ["a"+ letters[l]] # to get 52 instead of 26 individual letters
    	######################## Calibration############
    	if os.path.exists(calname2):
    		calexists = True
    		print("Cal File Exists - drawing points")
    		with open(calname2) as cal:
    				reader = csv.reader(cal)
    				row1 = next(reader)
    				row2 = next(reader)
    				row2 = str(row2)
    				row2= row2.replace("['","")
    				row2= row2.replace("']","")
    				row2= row2.split(";")
    				for i in range(len(row2)):
    					row2[i] = int(row2[i])
    				calpoints = row2
    				ret, frame = self.cam.read()
    				height, width, channels = frame.shape
    				new = np.zeros((height,width), np.uint8)
    				cutout = cv2.bitwise_and(frame,frame,mask = new)
    				del(row1,row2,cal,reader,height, width, channels,ret, frame)
    	else:
    		calexists = False
    	if batch:
    		batch = 1
    	while calibrate:
    		if fc == 0:
    			ret, frame = self.cam.read()
    			fc +=1
    			height, width, channels = frame.shape
    			new = np.zeros((height,width), np.uint8)
    		frame2 = frame.copy()
    		cv2.setMouseCallback(windowname,mousePosition)
    		cv2.line(frame2,(xmouse-200,ymouse),(xmouse+200,ymouse),(100,0,0),2)
    		cv2.line(frame2,(xmouse,ymouse-200),(xmouse,ymouse+200),(100,0,0),2)
    		if add_remove_pt:
    #				print("point added")
    				pa= True
    				calpoints += pt
    				add_remove_pt = False
    #				print(pt)
    #		if calpoints != () and len(calpoints) <3:
    #			cv2.line(frame2,(calpoints[0],calpoints[1]),(calpoints[0]+100,calpoints[1]),(0,0,20),1)
    #			cv2.line(frame2,(calpoints[0],calpoints[1]),(calpoints[0],calpoints[1]+100),(0,0,20),1)
    		if len(calpoints)%4 >0:
    			cv2.line(frame2,(calpoints[len(calpoints)-2],calpoints[len(calpoints)-1]),(calpoints[len(calpoints)-2]+100,calpoints[len(calpoints)-1]),(0,0,20),1)
    			cv2.line(frame2,(calpoints[len(calpoints)-2],calpoints[len(calpoints)-1]),(calpoints[len(calpoints)-2],calpoints[len(calpoints)-1]+100),(0,0,20),1)
    		if len(calpoints) >2:
    			circletimer = 1 # trigger to run circle drawing only every fourth point
    			pointstore=[]
    			pointtreat=""
    			pointheight=[]
    			pointwidth=[]
    			circlecount= 0
    			for points in calpoints:
    				pointstore = pointstore +[points]
    				
    				if circletimer == 4:
    					circlecount += 1
    					if (pointstore[2]-pointstore[0])>0:
    						cropped = cutout[pointstore[1]:pointstore[3],pointstore[0]:pointstore[2]]
    						red = cropped[:,:,2]	# extract redchannel
    						cropped2 = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    						cropped2= cv2.absdiff(cropped2, red) 	# difference of red channel to picture
    						pointheight=pointheight+[pointstore[3]-pointstore[1]]
    						pointwidth=pointwidth + [(pointstore[2]-pointstore[0])]
    						if np.amax(cropped2) > 20 and Colordetection:
    							cv2.circle(frame2,(int(pointstore[0]+(pointstore[2]-pointstore[0])/2),int(pointstore[1]+(pointstore[3]-pointstore[1])/2)),int((pointstore[2]-pointstore[0])/2),(0,0,200),2) # red circle
    							cv2.putText(frame2,str(circlecount),(int(pointstore[0]+(pointstore[2]-pointstore[0])/2),int(pointstore[1]+(pointstore[3]-pointstore[1])/2)), font, 1, (0,0,200), 2)
    							pointtreat = pointtreat + "t" # annotate last two coordinates as treatment
    						else:
#    							print(pointstore)
#    							print((pointstore[0]+(pointstore[2]-pointstore[0])/2,pointstore[1]+(pointstore[3]-pointstore[1])/2))
    							cv2.circle(frame2,(int(pointstore[0]+(pointstore[2]-pointstore[0])/2),int(pointstore[1]+(pointstore[3]-pointstore[1])/2)),int((pointstore[2]-pointstore[0])/2),(200,0,0),2)# blue circle
    							cv2.putText(frame2,str(circlecount),(int(pointstore[0]+(pointstore[2]-pointstore[0])/2),int(pointstore[1]+(pointstore[3]-pointstore[1])/2)), font, 1, (200,0,0), 2)
    							pointtreat = pointtreat + "c" # annotate last two coordinates as control
    						cv2.circle(new,(int(pointstore[0]+(pointstore[2]-pointstore[0])/2),int(pointstore[1]+(pointstore[3]-pointstore[1])/2)),int((pointstore[2]-pointstore[0])/2),(255,255,255),-1) #mask circle
    				circletimer +=1
    				if circletimer ==5:
    					circletimer = 1
    					pointstore=[]
    #		mask_inv = cv2.bitwise_not(new)
    		cutout = cv2.bitwise_and(frame,frame,mask = new)
    		cv2.imshow(windowname,frame2)
    #		cv2.imshow("cutout",cutout)
    ########################if not batch ################
    		if batch:
    			sn2 = savename[:-4] + str(fc) + '.png'
    			cv2.imwrite(sn2,frame2)
    			if batch == 2:
    				calibrate = False
 #   				show = False
    			batch += 1
    		if not batch:
    			ch = cv2.waitKey(10) % 0x100
    			if ch == 27:
    				sn2 = savename[:-4] + str(fc) + '.png'
    				cv2.imwrite(sn2,frame2)
    				calibrate = False
    			if 32 <= ch and ch < 128:
    				cc = chr(ch).lower()	
    				if cc == 'o':
    					sn2 = savename + str(fc) + '.png'
    					cv2.imwrite(sn2,frame2)
    					print("picture taken")
    					del(sn2)
    				if cc == 'd':
    					calpoints = calpoints[:len(calpoints)-2]
    				if cc == 's': # make elipses by 2 pixels smaller
    				  iteration = 1
    				  for points in calpoints:
    				    if iteration%4<3: 
    				      calpoints[iteration-1] = calpoints[iteration-1]+2
    				    if iteration%4>2:    	
    				      calpoints[iteration-1] = calpointss[iteration-1]-2
    				if cc == 'b': # make elipses by 2 pixels smaller
    				  iteration = 1
    				  for points in calpoints:
    				    if iteration%4<3: 
    				      calpoints[iteration-1] = calpoints[iteration-1]-2
    				    if iteration%4>2:    	
    				      calpoints[iteration-1] = calpointss[iteration-1]+2
    				      
    #	print "pointtreat", pointtreat
    #	print type(pointtreat)
    	fc=0	
    	calpointsw = str(calpoints)
    	calpointsw = calpointsw.replace(",",";")
    	calpointsw = calpointsw.replace("(","")
    	calpointsw = calpointsw.replace(")","")
    	calpointsw = calpointsw.replace("[","")
    	calpointsw = calpointsw.replace("]","")
    	text_file2 = open(calname, "w")
    	text_file2.write("#Calpoints" + "\n")
    	text_file2.write(calpointsw + "\n")
    	text_file2.close()
    #	if log:
    #		for i in xrange(len(pointtreat)):
    #		 	fileheader  = fileheader  + str(i) + "-" +pointtreat[i] + "; " + "\n"
    #		text_file.write(fileheader + "\n")
    #		text_file3.write(fileheader + "\n")
    	del(calpointsw, calibrate, channels, circletimer, cropped, cropped2, cutout, frame, frame2, image,new,red,circlecount)
    #	print(dir())
    #	print "max pointheight"
    #	print max(pointheight)
    #	print "pointswidthts"
    #	print pointwidth
    	calpics = dict()
    	frameaddspics = dict()
    	frameaddspics100 = dict()
    	calpicsold = dict()
    	# diffpics = dict()
    	# diffpicsold = dict()
    	averages = dict()
    	averages100 = dict()
    	# addup = dict()
    	# maxvalues = dict()
    	# accweightpics = dict()
    	# accweightpicsold = dict()
    	acc2 = False	# merge over one two seconds
    	isizet = 0
    	isizec = 0
    	alpha = 0.5 	# alpha is the new picture ## if alpha is 0.5 you see smooth behavior, if alpha is 0.3 you se only flicker if animal is mooving
    			# the later could help identify real speed when adding up mean grey value over a time window
    	beta = 1-alpha
    	original = True
    	originalold = True
    	thresh = False #### new with THRESH2
    	#####ÜBERARBEITEN!!!#######
    	# text_file.write("# Version:" + str(Version) + " Start Options: Merging:" + str(acc2) + ", alpha:" + str(alpha) + ",Thresholding:" + str(thresh)+ ", " + str(THRESH) + "\n")
    	# text_file3.write("# Version:" + str(Version) + " Start Options: Merging:" +  str(acc2) + ", alpha:" + str(alpha) + ",Thresholding:" + str(thresh)+ ", " + str(THRESH) + "\n")
    	# text_file4.write("# Version:" + str(Version) + " Start Options: Merging:" +  str(acc2) + ", alpha:" + str(alpha) + ",Thresholding:" + str(thresh)+ ", " + str(THRESH) + "\n")
    	text_file.write("# Version:" + str(Version) + " Start Options: Blurrate:" + str(BLUR) + ", Background Substraction:" + str(BACKGROUNDSUB) + ",Thresholding: " + str(THRESH2) + "\n")
    	text_file3.write("# Version:" + str(Version) + " Start Options: Blurrate:" + str(BLUR) + ", Background Substraction:" + str(BACKGROUNDSUB) + ",Thresholding: " + str(THRESH2) + "\n")
#    	text_file4.write("# Version:" + str(Version) + " Start Options: Blurrate:" +  str(acc2) + ", alpha:" + str(alpha) + ",Thresholding: " + str(THRESH2) + "\n")
#####################
    	if log:
    		for i in range(len(pointtreat)):
    		 	fileheader  = fileheader  + str(i) + "-" +pointtreat[i] + "; "
    		text_file.write(fileheader + "\n")
    		text_file3.write(fileheader + "\n")
#    		text_file4.write(fileheader + "\n")
    #	print(dir())
    	fgbg2 = cv2.createBackgroundSubtractorMOG2()
#    	print(skip)
    	while True:
    		ret, frame = self.cam.read()
    		if skip >0:
    		  if fc <= skip:
    		    print(str(fc) + " skipped")
    		    fc = fc + 1
    		    continue
    		  if fc > skip:
    		    fc = 0
    		    skip = False
    		    fcmax = fcmax - skip
#    		print(fc + skip)
    		if BACKGROUNDSUB:
    		  if BLUR:
    		    frame = cv2.blur(frame, (BLUR,BLUR))
    		  fgmask2 = fgbg2.apply(frame,learningRate = -1)
    		  ret, fgmask2 = cv2.threshold(fgmask2, 0, 255, cv2.THRESH_BINARY)
    		  if fc < 2:

    		    height, width = frame.shape[:2]
    		    frameadd = np.zeros((height, width), np.uint8)
    		    frameadd100 = np.zeros((height, width), np.uint8)
    		  if fc % 100 == 0:
    		    frameadd100 = np.zeros((height, width), np.uint8)
    		  if fc > 1:  
    		    frameadd = cv2.add(frameadd,fgmask2)
    		    frameadd100 = cv2.add(frameadd100,fgmask2)
#    		    frame = frameadd
    		    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    		if fc == 0:
    			height, width, channels = frame.shape
    			percold = -1
    		if fc < 2:
    			new2 = np.zeros((height,width), np.uint8)
    		if fc == 3:
    			height2 = max(pointheight)*3
    			width2 = max(isizec,isizet)
    		if original and original != originalold:
    			height2 = max(pointheight)*3
    			width2 = max(isizec,isizet)
    			originalold = original
    		if not original and original != originalold:
    			height2 = max(pointheight)*3
    			width2 = max(isizec,isizet)
    			originalold = original		
    		if fc > 2:
    			new2 = np.zeros((height2,width2), np.uint8)
    		for i in range(int(len(calpoints)/4)):
    			new = np.zeros((height,width), np.uint8)
    			p = i* 4
    #			print "i",i
    			cv2.circle(new,(int(calpoints[p]+(calpoints[p+2]-calpoints[p])/2),int(calpoints[p+1]+(calpoints[p+3]-calpoints[p+1])/2)),int((calpoints[p+2]-calpoints[p])/2),(255,255,255),-1)
    			cutout =cv2.bitwise_and(frame,frame,mask = new)
    			cutout =cutout[calpoints[p+1]:calpoints[p+3],calpoints[p]:calpoints[p+2]]
    			# print("cutout average brightness")
    			# print(np.average(np.average(cutout,axis=0)))
    			# print("new shape")
    			# print(new.shape)
    			frameadds = cv2.bitwise_and(frameadd,frameadd,mask = new)
    			frameadds = frameadds[calpoints[p+1]:calpoints[p+3],calpoints[p]:calpoints[p+2]]
    			frameadds100 = cv2.bitwise_and(frameadd100,frameadd100,mask = new)
    			frameadds100 = frameadds100[calpoints[p+1]:calpoints[p+3],calpoints[p]:calpoints[p+2]]
    			# print("frameadds average brightness")
    			# print(np.average(np.average(frameadds,axis=0)))
    			# print("Shape3")
    			# print("fc: " + str(fc))
    			# print(i)
    			# print("cutout")
    			# print(cutout.shape)
    			# print("frameadds")
    			# print(frameadds.shape)
    			if BRIGHTEN:
    			  brightness = np.sum(cutout) / (255 * height * width)
    			  # print("i "+str(i))    			
    			  # print("Brightness: ", brightness)
    			  minimum_brightness = 0.05 # was 0.66
    			  ratio = brightness / minimum_brightness
    			  # print(ratio)
    			  if ratio < 1:
    			    cutout = cv2.convertScaleAbs(cutout, alpha = 1 / ratio, beta = 0)

    			if BLUR and not BACKGROUNDSUB:
    			  cutout= cv2.blur(cutout, (BLUR,BLUR))
    			if CLAHE:
    			  cutout = clahe(cutout)
#### interactivethreshold
    			if fc == 10 and THRESHINTERACTIVE:
    			  def nothing(x):
    			    pass
    			  cv2.namedWindow('Interactive_Threshold')
    			  # create trackbars for color change
    			  cv2.createTrackbar('Thresh','Interactive_Threshold',90,255,nothing)
    			  cv2.createTrackbar('Max','Interactive_Threshold',255,255,nothing)
    			  switch = '0 : OFF \n1 : ON'
    			  #cv2.createTrackbar(switch, 'image',0,1,nothing)
    			  Thresh=90
    			  Max=255
    			  # threshold to binary
    			  threshf = cv2.threshold(cutout, 90, 255, cv2.THRESH_BINARY)[1]
    			  while(1):
    			    cv2.imshow('Interactive_Threshold',threshf)
    			    threshf = cv2.threshold(cutout, Thresh, Max, cv2.THRESH_BINARY)[1]
    			    k = cv2.waitKey(1) & 0xFF
    			    if k == 27:
    			      break 
    			    Thresh = cv2.getTrackbarPos('Thresh','Interactive_Threshold')
    			    Max = cv2.getTrackbarPos('Max','Interactive_Threshold')    		
#### interactivethreshold end    
    			if THRESH2:
    			  cutout = cv2.threshold(cutout, THRESH2, 255, cv2.THRESH_BINARY)[1]
    		#	cv2.imshow("cutout",cutout)
    #			print cutout
    #			calpics.update=({i:cutout})
    	#		del(new,cutout)
    #			if(pointtreat[i]=="t"):
    ################################UNCOMMENTED 13.01.22			
    			if fc == 0:
    # 				calpics.update({i:cv2.blur(cutout, (3,3))})
    # 				if not BACKGROUNDSUB:
    # 				  calpics.update({i:cv2.cvtColor(calpics[i], cv2.COLOR_BGR2GRAY)})
    # 				calpicsold.update({i:calpics[i]})
    # 				diffpics.update({i:calpics[i]})
    # 				diffpicsold.update({i:calpics[i]})
    # 				accweightpics.update({i:calpics[i]})
    # 				accweightpicsold.update({letters[i]:calpics[i]})
    # 				accweightpicsold.update({letters[i]+"1":calpics[i]})
    # 				accweightpicsold.update({letters[i]+"2":calpics[i]})
    # 				accweightpicsold.update({letters[i]+"3":calpics[i]})
    # 
    # #				for mod in range(1,4): # 1:59
    # #					accweightpicsold.update({(mod*(len(pointtreat)+5)+i):calpics[i]}) # mod = fc%60 und i = observationindex
    			  if i == 0:
    			    averages.update({"fc":[]})
    			    averages.update({"fc":averages["fc"]+[fc]})
    			    averages100.update({"fc":[]})
    			    averages100.update({"fc":averages100["fc"]+[fc]})
    			    # addup.update({"fc":[]})
    			    # maxvalues.update({"fc":[]})
    			  averages.update({i:[]})
    			  averages100.update({i:[]})
    			  # addup.update({i:0})
    			  # addup.update({letters[i]:[]})
    			  # maxvalues.update({i:[]})
    			  # maxvalues.update({letters[i]:[]})
#    			if fc > 0 and (fc+1)% 60 != 0: #####warum sollte das alle Minute nicht passieren?'######
    			if fc > 0:
    #				print "length of pointreat is:", len(pointtreat)
    #				print "fc:",fc," i:",i,"so:", (len(pointtreat)+5)+i,2*(len(pointtreat)+5)+i,3*(len(pointtreat)+5)+i
    				calpics.update({i:cutout})
    				frameaddspics.update({i:frameadds})
    				frameaddspics100.update({i:frameadds100})
    ##########################Uncommented 13.1.22
    # 				if not BACKGROUNDSUB:
    # 				  calpics.update({i:cv2.cvtColor(cutout, cv2.COLOR_BGR2GRAY)})
    # 				  diffpics.update({i:cv2.absdiff(calpics[i], calpicsold[i])})
    # 				calpicsold.update({i:calpics[i]})
    # #					grey2b = greyb
    				if i == 0:
    				  averages.update({"fc":averages["fc"]+[fc]})
    				  averages100.update({"fc":averages100["fc"]+[fc]})
    # 				if fc > 1:
    # 					accweightpics.update({i:np.uint8(alpha*(diffpics[i])+beta*(diffpicsold[i]))})
    # 				##################### code with letters instead of numbers!!!!
#    				if acc2 and fc >1:
    # 					accweightpicsold.update({letters[i]+"2": 	# old 2 will become
    # np.uint8(alpha*(accweightpicsold[letters[i]+"2"])+ 					# old 2 merged with
    # beta*(accweightpicsold[letters[i]+"3"]))}) 						# old 3 merged with
    # 
    # 					accweightpicsold.update({letters[i]+"1": 	# old 1 will become
    # np.uint8(alpha*(accweightpicsold[letters[i]+"1"])+ 					# old 1 merged with
    # beta*(accweightpicsold[letters[i]+"2"]))}) 						# old 2
    # 
    # 					accweightpicsold.update({letters[i]: 				# old will become
    # np.uint8(alpha*(accweightpics[i])+ 								# new merged with
    # beta*(accweightpicsold[letters[i]+"1"]))}) 						# old 1
    # 
    # 					accweightpics.update({i: 				# new 1
    # accweightpicsold[letters[i]]})										# will become old 1
    # 
    # 					accweightpicsold.update({letters[i]+"3":	# old 3
    # accweightpicsold[letters[i]+"2"]})									# will become 2
    # 
    # 					accweightpicsold.update({letters[i]+"2":	# old 2
    # accweightpicsold[letters[i]+"1"]})										# will become old 1
    # 					accweightpicsold.update({letters[i]+"1":	# old 1
    # accweightpicsold[letters[i]]})
    # ##					accweightpicsold.update({letters[i]:		# old will become
    # ##accweightpics[i]}) # change places								# new 1
    # #					accweightpics.update({i:accweightpicsold[i]})
    # 
    # #				averages.update({i:averages[i]+[(np.average(np.average(diffpics[i],axis=0)))]})
    # 				if thresh:
    # #					accweightpics.update({i:cv2.blur(accweightpics[i], (3,3))})
    # 					accweightpics.update({i:cv2.threshold(accweightpics[i], THRESH, 255, cv2.THRESH_BINARY)[1]})
    # 
    				
    			if log and fc >0:
      					# print ("i: "+str(i))
      					# print(np.average(np.average(frameaddspics[i],axis=0)))
      					# averages.update({i:[round(np.average(np.average(frameaddspics[i],axis=0)),2)]})
      					# averages100.update({i:[round(np.average(np.average(frameaddspics100[i],axis=0)),2)]})
      					averages.update({i:averages[i]+[round(np.average(np.average(frameaddspics[i],axis=0)),2)]})
      					averages100.update({i:averages100[i]+[round(np.average(np.average(frameaddspics100[i],axis=0)),2)]})
#       					if (fc+1) % 100 == 0: 
#       					  averages100.update({i:averages100[i]+[(np.average(np.average(frameaddspics100[i],axis=0)))]})
# #      					print(averages[i])
      # 					if not (fc)% 30 == 0:
      # 						addup.update({i:addup[i]+ np.average(np.average(frameaddspics[i],axis=0))})
      # 					if (fc)% 30 == 0:
      # 						addup.update({letters[i]:addup[letters[i]]+[addup[i]]})
      # 						maxvalues.update({i:maxvalues[i]+ [addup[i]]})
      # 						if i == 0:
      # #							print "writing fc in addup at fc:", fc
      # 							addup.update({"fc":addup["fc"]+[fc]})
      # 						addup.update({i:0})
      # 					if (fc)% 300 == 0 and fc > 300:
      # 						if i == 0:
      # 							maxvalues.update({"fc":maxvalues["fc"]+[fc]})
      # 						maxvalues.update({letters[i]:maxvalues[letters[i]]+[max(maxvalues[i])]})
      # #						print "300! i:", i, " fc:", fc
      # #						print maxvalues["fc"]
      # #						print "maxvalues (raw data points):", maxvalues[i]
      # #						print "maxvalues letters (max data points):", maxvalues[letters[i]]
      # 						maxvalues.update({i:[]})
      # 					if fc == 300:
      # 						maxvalues.update({i:[]})
    #						print addup
    #				(thresh, im_bw) = cv2.threshold(accweightpics[i], 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #				print fc, i, thresh
    				# accweightpics.update({i:cv2.threshold(accweightpics[i], 20, 255, cv2.THRESH_BINARY)[1]})
    				# accweightpics.update({i:cv2.blur(accweightpics[i], (3,3))})
    #				print "fc", fc, "modulus", fc%60
    				
    #				accweightpicsold.update({i:accweightpics[i]})
    				# diffpicsold.update({i:diffpics[i]})
    #				for p in range(1,60): # 1:59
    #					accweightpicsold.update({((p+1)*(len(pointtreat)+5))+i:accweightpicsold[(p*(len(pointtreat)+5))+i]}) # bigger picold numbers are older pics
    #				accweightpicsold.update({i:accweightpics[i]})
    #				accweightpicsold.update({(2*(len(pointtreat)+5)+i):((len(pointtreat)+5)+i)})
    					
    					
    #				del(cutout)
    #				print i
    #				print "size", diffpics[i].shape
    ### abhier use i size control and isize treatment to sumup width of pictures and shift these				#isize
    			if fc > 1 and (show or ((fc+1)>=fcmax)) and original:
    				if i == 0:
    					isizet = 0
    					isizec = 0
    #					addupc= 0
    #					addupt = 0
    					
    				if pointtreat[i]=="c":
    					if not BACKGROUNDSUB:
    					  waste, widthc = diffpics[i].shape
    					else:
#    					   print(calpics[i].shape)
    					   waste, widthc = calpics[i].shape
    					   # print(calpics[i].shape)
    					   # print(accweightpics[i].shape[0:2])
    					   # print(new2[max(pointheight)*2-pointheight[i]:max(pointheight)*2,isizec-widthc:isizec].shape)
    					isizec += widthc
    					# print(fc)
    					# print(((fc+1)>=fcmax))
    					if fc >2 and (show or ((fc+1)>=fcmax)):
    						# print("Shapes2")
    						# print(calpics[i].shape)
    						# print(frameaddspics[i].shape)
    						new2[max(pointheight)-pointheight[i]:max(pointheight),isizec-widthc:isizec]= calpics[i]
    						new2[max(pointheight)*2-pointheight[i]:max(pointheight)*2,isizec-widthc:isizec]= frameaddspics[i]
    						new2[max(pointheight)*3-pointheight[i]:max(pointheight)*3,isizec-widthc:isizec]= frameaddspics100[i]
    						cv2.putText(new2,str(i+1),(isizec-widthc,50), font, 1, (200,0,0), 2)
    						cv2.line(new2,((isizec-widthc),0),((isizec-widthc),height2),(200,0,0),2)
    #						cv2.putText(new2,str(letters[i]),(isizec-widthc,int(max(pointheight)*1.2)), font, 1, (200,0,0), 2)
    						if fc > 2:
#    							print(averages[i])
    							cv2.putText(new2,str(averages[i][averages["fc"].index(fc)-1]),(isizec-widthc,int(max(pointheight)*1.2)), font, 1, (200,0,0), 2)
    							cv2.putText(new2,str(averages100[i][averages100["fc"].index(fc)-1]),(isizec-widthc,int(max(pointheight)*2.2)), font, 1, (200,0,0), 2)
    
    				# else:
    				# 	waste, widtht = diffpics[i].shape	
    				# 	isizet += widtht	
    				# 	if fc >2:	
    				# 		new2[height2-max(pointheight):height2-max(pointheight)+pointheight[i],isizet-widtht:isizet]= calpics[i]
    				# 		new2[height2-max(pointheight)*2:height2-max(pointheight)*2+pointheight[i],isizet-widtht:isizet]= frameaddspics[i]
    				# 		if fc > 60:
    				# 			cv2.putText(new2,str(round(addup[letters[i]][len(addup[letters[i]])-1],2)),(isizet-widtht,int(height2-max(pointheight)*1.1)), font, 1, (200,0,0), 2)
    			if fc > 1 and show and not original:
    				if i == 0:
    					isizet = 0
    					isizec = 0
    				if pointtreat[i]=="c":		
    					waste, widthc = diffpics[i].shape	
    					isizec += widthc
    					if fc >2:		
    						new2[max(pointheight)-pointheight[i]:max(pointheight),isizec-widthc:isizec]= accweightpics[i]
    				else:
    					waste, widtht = diffpics[i].shape	
    					isizet += widtht	
    					if fc >2:	
    						new2[height2-max(pointheight):height2-max(pointheight)+pointheight[i],isizet-widtht:isizet]= accweightpics[i]
    #			print "fc",fc,"isizet",isizet,"isizec",isizet
    #		if fc >5 and fc%60 != 0 and (fc-1)%60 != 0 and (fc-2)%60 != 0:
    #			average_colorre = np.average(np.average(diffreb,axis=0)) # calculate average grey value
    #			average_colorli = np.average(np.average(difflib,axis=0))
    #			print fc
    #			print average_colorli
    #		if show:
    #			if fc >5:
    #				if fc < width:
    #					ptsli = ptsli + (int(fc),height-25-int(average_colorli*100)) # generate vector with framecount and average value
    #					ptsre = ptsre + (int(fc),height-5-int(average_colorre*100))
    #				if fc == width:
    #					### li
    #					ptsli = ptsli + (int(fc),height-25-int(average_colorli*100)) 	# generate vector with framecount and average value
    #													# with fc/10 if plot gets out of the window
    ###############				fcs = [x/10 for x in range(0,fc)]
    #					fcs = range(0,fc)
    #					ptslib = ptsli[1::2]
    #					print len(fcs)
    #					print len(ptslib)
    					
    #					ptslic = (0,sum(ptslib[0:10])/10)
    #					for i in range(1,(fc/10)-1):
    #						print i
    #						p = i*10
    #						ptslic = ptslic + (int(fc),sum(ptslib[p:p+10])/10)
    				#	[x/10 for x in range(0,fc)]
    #					print ptsli
    #					ptsli = ptslic
    #					for i in range(1,len(ptslib)):
    #						ptsli = ptsli + (fcs[i],ptslic[i])
    					# re
    #					ptsre = ptsre + (int(fc),height-5-int(average_colorre*100))
    #					ptsreb = ptsre[1::2]
    #					ptsre = ()
    #					for i in range(1,len(ptsreb)):
    #						ptsre = ptsre + (fcs[i],ptsreb[i])
    #				if fc >	width:		
    #					ptsli = ptsli + (int(fc/10),height-25-int(average_colorli*100))
    #					ptsre = ptsre + (int(fc/10),height-5-int(average_colorre*100))
    #				print pts
    #			if fc>6 and ptsli !=[]:
    #				cv2.polylines(new, [[0,0],[255,255]], True, (255,255,255))
    #				print len(pts)
    #				cv2.putText(new,"left",(0,height-60), font, 0.5, (200,0,0), 1)
    #				cv2.putText(new,"right",(0,height-5), font, 0.5, (250,0,0), 1)
    #				for i in range(0,(len(ptsli)/2)-2):
    #					p = i *2
    #					print "i",i
    #					print (ptsli[p],ptsli[p+1])
    #					print (ptsli[p+2],ptsli[p+3])
    #					cv2.line(new,(ptsli[p],ptsli[p+1]),(ptsli[p+2],ptsli[p+3]),(200,0,0),1)
    #				for i in range(0,(len(ptsli)/2)-2):
    #					p = i *2
    				
    #					cv2.line(new,(ptsre[p],ptsre[p+1]),(ptsre[p+2],ptsre[p+3]),(255,0,0),1)
    ##				print pts
    #			cv2.putText(new,str(fc),(width-30,height-3), font, 0.5, (200,255,155), 1)
    #			if fc%60 == 0:
    #				cv2.putText(new,"60!",(width-60,height-3), font, 0.5, (255,100,100), 1)
    #		print diffpics.keys()
    #		if fc == 10:
    #			print "break"
    #			print accweightpicsold.keys()
    #			print accweightpics.keys()
    #			break
# #### interactivethreshold
#     		if fc == 10 and THRESHINTERACTIVE:
#     		  def nothing(x):
#     		    pass
#     		  cv2.namedWindow('Interactive_Threshold')
#     		  # create trackbars for color change
#     		  cv2.createTrackbar('Thresh','Interactive_Threshold',90,255,nothing)
#     		  cv2.createTrackbar('Max','Interactive_Threshold',255,255,nothing)
#     		  switch = '0 : OFF \n1 : ON'
#     		  #cv2.createTrackbar(switch, 'image',0,1,nothing)
#     		  Thresh=90
#     		  Max=255
#     		  # threshold to binary
#     		  threshf = cv2.threshold(new2, 90, 255, cv2.THRESH_BINARY)[1]
#     		  while(1):
#     		    cv2.imshow('Interactive_Threshold',threshf)
#     		    threshf = cv2.threshold(new2, Thresh, Max, cv2.THRESH_BINARY)[1]
#     		    k = cv2.waitKey(1) & 0xFF
#     		    if k == 27:
#     		      break 
#     		    Thresh = cv2.getTrackbarPos('Thresh','Interactive_Threshold')
#     		    Max = cv2.getTrackbarPos('Max','Interactive_Threshold')
#     		
# #### interactivethreshold end
    		if fc > 2:
    		  cv2.line(new2,(0,max(pointheight)),(width2,max(pointheight)),(200,0,0),2)
    		  cv2.line(new2,(0,max(pointheight)*2),(width2,max(pointheight)*2),(200,0,0),2)
    		  cv2.putText(new2,"original",(10,20), font, 1, (255,0,0), 2)
    		  cv2.putText(new2,"addup, blurrate "+str(BLUR),(10,max(pointheight)+30), font, 1, (255,0,0), 2)
    		  cv2.putText(new2,"100 frames, blurrate "+str(BLUR),(10,(max(pointheight)*2)+30), font, 1, (255,0,0), 2)
    						# new2[max(pointheight)*3-pointheight[i]:max(pointheight)*3,isizec-widthc:isizec]= frameaddspics100[i]
    						# cv2.putText(new2,str(i+1),(isizec-widthc,30), font, 1, (200,0,0), 2)
    		if fc>1 and (fc+1)% 60 != 0 and (fc)% 60 != 0:
    #			print fc, averages[0][averages["fc"].index(fc)-1]
    			if not show:
    				new2 = np.zeros((30,30), np.uint8)
    				perc = str(int((fc*1.0/fcmax*1)*100))+"%"
    				cv2.putText(new2,perc,(1,20), font, 0.5, (255,100,100), 1)
    				new2 = cv2.blur(new2, (1,1))
    				if (fc)% 59:
    					cv2.imshow(windowname,new2)
    			else:
    				cv2.imshow(windowname,new2)
    #			cv2.imshow(windowname,accweightpicsold[1])
    #			cv2.imshow("accweightpicsold",accweightpicsold[1])
#################Broken please repair#####
    			if log:
#    				print(fc)
    				savings = str(averages["fc"].index(fc)) + "; "
    				for i in range(len(pointtreat)):
    		 			savings  = savings  + str(averages[i][averages["fc"].index(fc)-1]) + "; "
    				text_file.write(savings+ "\n")
    				if (fc+1) %100 ==0:
    				  savings = str(averages100["fc"].index(fc)) + "; "
    				  for i in range(len(pointtreat)):
    				    savings  = savings  + str(averages100[i][averages100["fc"].index(fc)-1]) + "; "
    				  text_file3.write(savings+ "\n")
    #		print fc, " of ", fcmax
    # 		if log and (fc)% 30 ==0 and fc >30 and False:
    # #				print "saved text_file3"
    # 				savings = str(fc) + "; "
    # 				for i in range(len(pointtreat)):
    # 		 			savings  = savings  + str(addup[letters[i]][addup["fc"].index(fc)]) + "; "
    # 				text_file3.write(savings+ "\n")
    # 		if log and (fc)% 300 ==0 and fc >300:
    # 				savings = str(fc) + "; "
    # 				for i in range(len(pointtreat)):
    # 		 			savings  = savings  + str(maxvalues[letters[i]][maxvalues["fc"].index(fc)]) + "; "
    # 				text_file4.write(savings+ "\n")	
    # #				print "saved text_file4"	
    # #		cv2.imshow("1",diffpics[1])
    # #		cv2.imshow("2",diffpics[2])
    #		if fc == 100:
    #			text_file.close()
    #			break
    		fc += 1
     # or fc == 1:
    		ch = cv2.waitKey(WaitTime) % 0x100
    		if ch == 27:
    				text_file.close()
    				break
    		if 32 <= ch and ch < 128:
    				cc = chr(ch).lower()
    				if cc == 'a':	
    					if WaitTime == 1:
    							print("Stopped playback \n") 
    							WaitTime = 0
    					else:
    						print("Automated playback \n") 
    						WaitTime = 1
    				elif cc == 'v':
    					original = not original
    				elif cc == 's':
    					show = not show
    					Waittime = 0.01
    #				elif cc == 'm':
    #					acc2= not acc2
    #					if acc2:
    #						print "merge2"
    #					else:
    #						print "no merge"
    #				elif cc == 't':
    #					thresh= not thresh
    #					if thresh:
    #						print "threshold binary"
    #					else:
    #						print "threshold binary off"
    				else: 
    					cv2.waitKey(0)
    #		print fc
    #		print (int((fc*1.0/fcmax*1)*100)==5)
    		if (int((fc*1.0/fcmax*1)*100)%5==0):
    			perc = int((fc*1.0/fcmax*1)*100)
    			if perc== 100:
    				cv2.imwrite(picname,new2)
    				print(str(fc) + " of " + str(fcmax))
    				print("100%!")
    				print("done")
    #				text_file.write(str(averages))
    				text_file.close()
    				text_file3.close()
#    				text_file4.close()
    				break
    			if perc != percold:
    				print(str(int((fc*1.0/fcmax*1)*100)) + "% done")
    			percold = perc

#		print fc

####################################################### ###
#### MAIN ############################################# ###
####################################################### ###

def main():
    import sys
    global batch
    global show
    batch = False
    if len(sys.argv) == 1:
	    from tkinter import Tk
	    from tkinter.filedialog import askopenfilename

	    print("Please select video file to track")
	    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
	    srcname = askopenfilename() # show an "Open" dialog box and return the path to the selected file
	    print(srcname)
	    try: video_src = srcname
	    except: video_src = 0
	    if len(sys.argv) > 2:
			    
			    if sys.argv[2] == "show":
				    show = True
			    if sys.argv[2] == "2":
				    print("Show =2")
				    show = 2
			    if sys.argv[2] == "3":
				    print("Show =3")
				    show = 3

		
	    print(__doc__)
	    if batch == False:
    		App(video_src).run()
    		cv2.destroyAllWindows()
    else:
      if sys.argv[1] == "batch":
        from tkinter import Tk
        from tkinter.filedialog import askdirectory
        print("Please select video folders with videos and calfiles")
        Tk().withdraw()
        sourcedir=askdirectory()
        batch = True
        print("")
    #		print sourcedir
    #		for files in os.walk(sourcedir):  
    #			print files[2]
        for root, dirs, files in os.walk(sourcedir):
    #			print "root", root
    #			print "dirs", dirs
    #			print "files", files
          for srcname in files:
            if srcname.endswith(".txt") or srcname.endswith(".png"):
              pass
            else:
              srcname = root + "/"+ srcname
              print(srcname)
              try: video_src = srcname
              except: video_src = 0
              App(video_src).run()
              cv2.destroyAllWindows()
        print('\a'); print('\a'); print('\a')
      else:
        srcname = sys.argv[1]
        print(srcname)
    #	print "Got", srcname
    
        try: video_src = srcname
        except: video_src = 0
        if len(sys.argv) > 2:
          if sys.argv[2] == "show":
            show = True
          if sys.argv[2] == "2":
            print("Show =2")
            show = 2
          if sys.argv[2] == "3":
            print("Show =3")
            show = 3

		
    print(__doc__)
    # if batch == False:
    # 			App(video_src).run()
    # 			cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

