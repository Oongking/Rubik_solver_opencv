import kociemba
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils

L_rubik_blue   = np.array([100,35,120])
U_rubik_blue   = np.array([130,255,255])
L_rubik_red    = np.array([0,35,120])
U_rubik_red    = np.array([20,255,255])
L_rubik_green  = np.array([60,35,120])
U_rubik_green  = np.array([80,255,255])
L_rubik_yellow = np.array([35,35,120])
U_rubik_yellow = np.array([50,255,255])
L_rubik_orange = np.array([5,35,120])
U_rubik_orange = np.array([25,255,255])
L_rubik_white  = np.array([0,0,120])
U_rubik_white  = np.array([179,50,255])

#L_rubik_all  = np.array([0,0,0]) #need 'Mask = cv2.bitwise_not(Mask)'
#U_rubik_all  = np.array([179,255,91]) 

lower_color = np.array([0,0,0])
upper_color = np.array([0,0,0])
kernel = np.ones((50, 50), np.uint8)
block_location = [[0 for i in range(2)] for j in range(9)]
pattern = []
text_color = (0,0,0)
text_color_in = (0,255,255)
cx = 0
cy = 0
Full_pattern = ""

def find_hsv_value():
    global lower_color
    global upper_color
    l_h = cv2.getTrackbarPos("Low - H","HSV selection")
    l_s = cv2.getTrackbarPos("Low - S","HSV selection")
    l_v = cv2.getTrackbarPos("Low - V","HSV selection")
    u_h = cv2.getTrackbarPos("Upper - H","HSV selection")
    u_s = cv2.getTrackbarPos("Upper - S","HSV selection")
    u_v = cv2.getTrackbarPos("Upper - V","HSV selection")

    # lower_color = np.array([l_h,l_s,l_v])
    # upper_color = np.array([u_h,u_s,u_v])
    lower_color = np.array([0,0,0]) #for test
    upper_color = np.array([179,255,91]) #for test

def show_size(image_in):
    h, w, c = image_in.shape
    print('width:  ', w)
    print('height: ', h)
    print('channel:', c) #output type

def SortingPoint(Points):
    top_part = []
    mid_part = []
    bot_part = []

    # print('Points : ',Points)
    SortedY_Point = sorted(Points , key=lambda k: [k[1], k[0]])
    # print('Sorted Y Points : ',SortedY_Point)
    for x in range(9):
        if x < 3 :
            top_part.append(SortedY_Point[x])
        elif x < 6 :
            mid_part.append(SortedY_Point[x])
        elif x < 9 :
            bot_part.append(SortedY_Point[x])
    # print('Top part : ',top_part)
    # print('Mid part : ',mid_part)
    # print('Bot part : ',bot_part)
    top_part = sorted(top_part , key=lambda k: [k[0], k[1]])
    mid_part = sorted(mid_part , key=lambda k: [k[0], k[1]])
    bot_part = sorted(bot_part , key=lambda k: [k[0], k[1]])

    Sum_point = top_part + mid_part + bot_part
    # print('Sum : ',Sum_point)
    return Sum_point

def ShowPlot(images,titles):
    for i,_ in enumerate(images):
        plt.subplot(2,3,i+1),plt.imshow(imutils.opencv2matplotlib(images[i]))
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    plt.show()

def convert(s):
  
    # initialization of string to ""
    new = ""
  
    # traverse in the string 
    for x in s:
        new += x 
  
    # return string 
    return new

def get_color(HSV, position):
    HSVofPoint = []
    colorpattern = []

    for i,_ in enumerate(position):
        # print('HSV : ',HSV[position[i][1],position[i][0]])
        HSVofPoint.append(HSV[position[i][1],position[i][0]])
    for i,value in enumerate(HSVofPoint):
        if (L_rubik_blue[0] <= value[0] <= U_rubik_blue[0]) and (L_rubik_blue[1] <= value[1] <= U_rubik_blue[1]) and (L_rubik_blue[2] <= value[2] <= U_rubik_blue[2]):
            colorpattern.append('F') # F = Blue
        elif (L_rubik_red[0] <= value[0] <= U_rubik_red[0]) and (L_rubik_red[1] <= value[1] <= U_rubik_red[1]) and (L_rubik_red[2] <= value[2] <= U_rubik_red[2]):
            colorpattern.append('R') # R = Red
        elif (L_rubik_green[0] <= value[0] <= U_rubik_green[0]) and (L_rubik_green[1] <= value[1] <= U_rubik_green[1]) and (L_rubik_green[2] <= value[2] <= U_rubik_green[2]):
            colorpattern.append('B') # B = Green
        elif (L_rubik_yellow[0] <= value[0] <= U_rubik_yellow[0]) and (L_rubik_yellow[1] <= value[1] <= U_rubik_yellow[1]) and (L_rubik_yellow[2] <= value[2] <= U_rubik_yellow[2]):
            colorpattern.append('U') # U = Yellow 
        elif (L_rubik_orange[0] <= value[0] <= U_rubik_orange[0]) and (L_rubik_orange[1] <= value[1] <= U_rubik_orange[1]) and (L_rubik_orange[2] <= value[2] <= U_rubik_orange[2]):
            colorpattern.append('L') # L = Orange
        elif (L_rubik_white[0] <= value[0] <= U_rubik_white[0]) and (L_rubik_white[1] <= value[1] <= U_rubik_white[1]) and (L_rubik_white[2] <= value[2] <= U_rubik_white[2]):
            colorpattern.append('D') # D = White
      
    print('Position : ', position)
    print('color : ', colorpattern) 
    Full_pattern = convert(colorpattern)
    print('Full_pattern : ', Full_pattern) 
    print('Number of center :',len(colorpattern))
    return colorpattern, len(colorpattern)

def color_text(Points_location, pattern):

    for i,color in enumerate(pattern):
        # cv2.putText(img, "Centre",(Points_location[i][0]-25,Points_location[i][1]-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color,5)
        cv2.putText(img, "Color = "+ pattern[i], (Points_location[i][0]-38,Points_location[i][1]-12),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color,5)

        # cv2.putText(img, "Centre",(Points_location[i][0]-25,Points_location[i][1]-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color_in,2)
        cv2.putText(img, "Color = "+ pattern[i], (Points_location[i][0]-38,Points_location[i][1]-12),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color_in,2)
    
    Xpos = 1700
    ypos = 80
    cv2.putText(img, "F = Blue "    , (Xpos, ypos),cv2.FONT_HERSHEY_SIMPLEX,0.6,text_color,5)
    cv2.putText(img, "R = Red "     , (Xpos, ypos+20),cv2.FONT_HERSHEY_SIMPLEX,0.6,text_color,5)
    cv2.putText(img, "B = Green "   , (Xpos, ypos+40),cv2.FONT_HERSHEY_SIMPLEX,0.6,text_color,5)
    cv2.putText(img, "U = Yellow "  , (Xpos, ypos+60),cv2.FONT_HERSHEY_SIMPLEX,0.6,text_color,5)
    cv2.putText(img, "L = Orange "  , (Xpos, ypos+80),cv2.FONT_HERSHEY_SIMPLEX,0.6,text_color,5)
    cv2.putText(img, "D = White "   , (Xpos, ypos+100),cv2.FONT_HERSHEY_SIMPLEX,0.6,text_color,5)

    cv2.putText(img, "F = Blue "    , (Xpos, ypos),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,128,0),2)
    cv2.putText(img, "R = Red "     , (Xpos, ypos+20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(51,51,255),2)
    cv2.putText(img, "B = Green "   , (Xpos, ypos+40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    cv2.putText(img, "U = Yellow "  , (Xpos, ypos+60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
    cv2.putText(img, "L = Orange "  , (Xpos, ypos+80),cv2.FONT_HERSHEY_SIMPLEX,0.6,(51,153,255),2)
    cv2.putText(img, "D = White "   , (Xpos, ypos+100),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

def nothing(x):
    pass


path = r'D:\Oongking\program\Python study\imagepro\Rubik solver\data\shuffle (5).jpg'
image = cv2.imread(path)
img = imutils.rotate(image, 180)
blurred_img = cv2.GaussianBlur(img, (21,21),0)
hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)
yuv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2YUV)
# show_size(img)

cv2.namedWindow("HSV selection")
cv2.createTrackbar("Low - H","HSV selection", 0,179,nothing)
cv2.createTrackbar("Low - S","HSV selection", 0,255,nothing)
cv2.createTrackbar("Low - V","HSV selection", 0,255,nothing)
cv2.createTrackbar("Upper - H","HSV selection", 0,179,nothing)
cv2.createTrackbar("Upper - S","HSV selection", 0,255,nothing)
cv2.createTrackbar("Upper - V","HSV selection", 0,255,nothing)

o=0

while(1):
    
    find_hsv_value()

    mask = cv2.inRange(hsv_img, lower_color, upper_color)
    mask = cv2.bitwise_not(mask) #for all color
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for index,contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area>5000:
            # cv2.drawContours(img, contour,-1,(0,255,0),3) #check corner
            M = cv2.moments(contour)
            approx = cv2.approxPolyDP(contour,0.001*cv2.arcLength(contour, True), True)
            cv2.drawContours(img, [approx],0,(102,0,102),4)
            cv2.drawContours(img, [approx],0,(0,255,255),2)

            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])

            block_location[index] = [cx,cy]

            cv2.circle(img,(cx,cy),7,(0,0,0),-1)
            cv2.circle(img,(cx,cy),5,(0,255,255),-1)

            cv2.putText(img, "Centre",(cx-25,cy-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color,5)
            cv2.putText(img, "X = "+str(cx),(cx-30,cy+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color,5)
            cv2.putText(img, "Y = "+str(cy),(cx-30,cy+40),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color,5)

            cv2.putText(img, "Centre",(cx-25,cy-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color_in,2)
            cv2.putText(img, "X = "+str(cx),(cx-30,cy+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color_in,2)
            cv2.putText(img, "Y = "+str(cy),(cx-30,cy+40),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color_in,2)

    if o == 0:
        Sorted_location = SortingPoint(block_location)
        print('Sorted point : ',Sorted_location)
        pattern,Num_of_point = get_color(hsv_img, Sorted_location)
        color_text(Sorted_location, pattern)
        o = o +1

    # a = 'UUFUUFUUFRRRRRRRRRFFDFFDFFDDDBDDBDDBLLLLLLLLLUBBUBBUBB'
    # T = kociemba.solve(a)
    # cv2.putText(img, T, (800,200),cv2.FONT_HERSHEY_SIMPLEX,1,text_color_in,2)

    Show_image = cv2.resize(img,(1280,720))
    Show_blurred = cv2.resize(blurred_img,(1280,720))
    Show_mask = cv2.resize(mask,(1280,720))
    Show_HSV = cv2.resize(hsv_img,(1280,720))
    Show_YUV = cv2.resize(yuv_img,(1280,720))


    cv2.imshow("Original Image",Show_image)
    #cv2.imshow("Blurred Image",Show_blurred)
    cv2.imshow("Selection",Show_mask)
    cv2.imshow("HSV",Show_HSV)
    cv2.imshow("YUV",Show_YUV)
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
# print(lower_color)
# print(upper_color)
cv2.destroyAllWindows()

titles = ['Original Image','Blurred Image','Selection','HSV Image']
images = [img, Show_blurred, Show_mask,hsv_img]
ShowPlot(images, titles)
