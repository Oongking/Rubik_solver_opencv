import kociemba
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import time
from difflib import SequenceMatcher

# Ver1
# L_rubik_blue   = np.array([100,35,120])
# U_rubik_blue   = np.array([130,255,255])
# L_rubik_red    = np.array([0,35,120])
# U_rubik_red    = np.array([20,255,255])
# L_rubik_green  = np.array([60,35,120])
# U_rubik_green  = np.array([80,255,255])
# L_rubik_yellow = np.array([35,35,120])
# U_rubik_yellow = np.array([50,255,255])
# L_rubik_orange = np.array([5,35,120])
# U_rubik_orange = np.array([25,255,255])
# L_rubik_white  = np.array([0,0,120])
# U_rubik_white  = np.array([179,50,255])

# Ver2
L_rubik_blue   = np.array([100,80,100])
U_rubik_blue   = np.array([130,255,255])
L_rubik_green  = np.array([60,80,120])
U_rubik_green  = np.array([80,255,255])
L_rubik_yellow = np.array([20,35,120])
U_rubik_yellow = np.array([60,255,255])
L_rubik_white  = np.array([0,0,130])
U_rubik_white  = np.array([179,60,255])
L_rubik_red    = np.array([0,35,130])
U_rubik_red    = np.array([10,255,255])
L_rubik_orange = np.array([9,35,130])
U_rubik_orange = np.array([20,255,255])

Blue    = [255,0,0]
Red     = [0,0,255]
Green   = [0,255,0]
Yellow  = [0,255,255]
Orange  = [0,128,255]
White   = [255,255,255]

#L_rubik_all  = np.array([0,0,0]) #need 'Mask = cv2.bitwise_not(Mask)'
#U_rubik_all  = np.array([179,255,91]) 

# /////SET/////
# lower_color = np.array([0,0,0]) 
# upper_color = np.array([179,255,91])
lower_color = np.array([0,0,0])
upper_color = np.array([0,0,0])

kernel = np.ones((50, 50), np.uint8)
Ero_kernel = np.ones((15, 15), np.uint8)
block_location = []
pattern = []
text_color = (0,0,0)
text_color_in = (0,255,255)
BG_color = (0,0,0)
arrow_color = (0,255,0)

cx = 0
cy = 0
Full_face_color = ""
Face_Color_Pattern = [(0,0,0) for j in range(54)]
In_FaceVerify_T = 0
Diff_FaceVerify_T = 0
Out_FaceVerify_T = 0
Toggle = 0

Face = ['','','','','','']
face_num = 0
SumFace = ""
Face_step = 0
Root = ''

def find_hsv_value():
    global lower_color
    global upper_color

    l_h = cv2.getTrackbarPos("Low - H","HSV selection")
    l_s = cv2.getTrackbarPos("Low - S","HSV selection")
    l_v = cv2.getTrackbarPos("Low - V","HSV selection")
    u_h = cv2.getTrackbarPos("Upper - H","HSV selection")
    u_s = cv2.getTrackbarPos("Upper - S","HSV selection")
    u_v = cv2.getTrackbarPos("Upper - V","HSV selection")

    lower_color = np.array([l_h,l_s,l_v])
    upper_color = np.array([u_h,u_s,u_v])
    # lower_color = np.array([0,0,0]) #for all
    # upper_color = np.array([179,255,91]) #for all

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
    for x,_ in enumerate(SortedY_Point):
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

    top_part.clear()
    mid_part.clear()
    bot_part.clear()

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
    global Full_face_color
    HSVofPoint = []
    colorpattern = []

    for i,_ in enumerate(position):
        # print('HSV : ',HSV[position[i][1],position[i][0]])
        HSVofPoint.append(HSV[position[i][1],position[i][0]])
        
    for i,value in enumerate(HSVofPoint):
        if (L_rubik_blue[0] <= value[0] <= U_rubik_blue[0]) and (L_rubik_blue[1] <= value[1] <= U_rubik_blue[1]) and (L_rubik_blue[2] <= value[2] <= U_rubik_blue[2]):
            colorpattern.append('B') # B = Blue
        elif (L_rubik_red[0] <= value[0] <= U_rubik_red[0]) and (L_rubik_red[1] <= value[1] <= U_rubik_red[1]) and (L_rubik_red[2] <= value[2] <= U_rubik_red[2]):
            colorpattern.append('R') # R = Red
        elif (L_rubik_green[0] <= value[0] <= U_rubik_green[0]) and (L_rubik_green[1] <= value[1] <= U_rubik_green[1]) and (L_rubik_green[2] <= value[2] <= U_rubik_green[2]):
            colorpattern.append('G') # G = Green
        elif (L_rubik_yellow[0] <= value[0] <= U_rubik_yellow[0]) and (L_rubik_yellow[1] <= value[1] <= U_rubik_yellow[1]) and (L_rubik_yellow[2] <= value[2] <= U_rubik_yellow[2]):
            colorpattern.append('Y') # Y = Yellow 
        elif (L_rubik_orange[0] <= value[0] <= U_rubik_orange[0]) and (L_rubik_orange[1] <= value[1] <= U_rubik_orange[1]) and (L_rubik_orange[2] <= value[2] <= U_rubik_orange[2]):
            colorpattern.append('O') # O = Orange
        elif (L_rubik_white[0] <= value[0] <= U_rubik_white[0]) and (L_rubik_white[1] <= value[1] <= U_rubik_white[1]) and (L_rubik_white[2] <= value[2] <= U_rubik_white[2]):
            colorpattern.append('W') # W = White
        elif(1):
            colorpattern.append('(E)')

      
    # print('Position : ', position)
    # print('color : ', colorpattern) 
    Full_face_color = convert(colorpattern)

    # print('Full_face_color : ', Full_face_color) 
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

def Verify_HSV():
    cv2.namedWindow("HSV selection")
    cv2.createTrackbar("Low - H","HSV selection", 0,179,nothing)
    cv2.createTrackbar("Low - S","HSV selection", 0,255,nothing)
    cv2.createTrackbar("Low - V","HSV selection", 0,255,nothing)
    cv2.createTrackbar("Upper - H","HSV selection", 0,179,nothing)
    cv2.createTrackbar("Upper - S","HSV selection", 0,255,nothing)
    cv2.createTrackbar("Upper - V","HSV selection", 0,255,nothing)

def Face_verify(S_pattern,Num_P,Srt_L):
    global In_FaceVerify_T 
    global Out_FaceVerify_T
    global Diff_FaceVerify_T
    global Face
    global face_num 
    global Toggle
    global Face_step
    In_FaceVerify_T = int(round(time.time() * 1000))
    # print('In_FaceVerify_T : ',In_FaceVerify_T)
    # print('Out_FaceVerify_T : ',Out_FaceVerify_T)
    if Num_P==9 and "(E)" not in S_pattern:
        
        print('In Face_Verify')
        # print('S_pattern :', S_pattern)
        # print('Face[face_num-1] :', Face[face_num-1])
        if Face_step == 1:
            Sign_BTW_Face(Srt_L, Face_step)
            cv2.putText(img, "Same Face . . .", (900,120),cv2.FONT_HERSHEY_SIMPLEX,1,text_color,5)
            cv2.putText(img, "Same Face . . .", (900,120),cv2.FONT_HERSHEY_SIMPLEX,1,text_color_in,2)
            

        if S_pattern == Face[face_num-1]:
            Toggle = 1
            cv2.putText(img, "Same Face . . .", (900,120),cv2.FONT_HERSHEY_SIMPLEX,1,text_color,5)
            cv2.putText(img, "Same Face . . .", (900,120),cv2.FONT_HERSHEY_SIMPLEX,1,text_color_in,2)

            Sign_BTW_Face(Srt_L, Face_step)
            

        else:

            if face_num == 1 or face_num == 2 or face_num == 3:
                if S_pattern[4] != Face[face_num-1][4]:
                    Toggle = 0

            elif face_num == 4:
                if S_pattern[4] != Face[face_num-1][4] and S_pattern[4] != Face[0][4]:
                    Toggle = 0
                    Face_step = 0
                if S_pattern[4] == Face[0][4]:
                    Face_step = 1

            elif face_num == 5:

                if S_pattern[4] != Face[face_num-1][4] and S_pattern[4] != Face[0][4]:
                    Toggle = 0
                if S_pattern[4] == Face[0][4] or S_pattern[4] == Face[1][4] or S_pattern[4] == Face[2][4] or S_pattern[4] == Face[3][4]:
                    Face_step = 1
    
        if face_num == 6:
                Sign_BTW_Face(Srt_L, Face_step) 
                if S_pattern == Face[0]:
                    face_num += 1 

        elif Num_P==9 and "(E)" not in S_pattern and Toggle == 0:
            Diff_FaceVerify_T = In_FaceVerify_T - Out_FaceVerify_T
            cv2.putText(img, "Verifying . . . Time : " + str(Diff_FaceVerify_T), (800,120),cv2.FONT_HERSHEY_SIMPLEX,1,text_color,5)
            cv2.putText(img, "Verifying . . . Time : " + str(Diff_FaceVerify_T), (800,120),cv2.FONT_HERSHEY_SIMPLEX,1,text_color_in,2)
            Face_step = 0
            # print(Diff_FaceVerify_T)

            if Diff_FaceVerify_T > 1000 :
                Store_face(S_pattern)
                # print('Store_face : ',Diff_FaceVerify_T)
        
        
    else:
        Out_FaceVerify_T = In_FaceVerify_T
        if Num_P!=9:
            cv2.putText(img, "Face not Full", (900,150),cv2.FONT_HERSHEY_SIMPLEX,1,text_color,5)
            cv2.putText(img, "Face not Full", (900,150),cv2.FONT_HERSHEY_SIMPLEX,1,text_color_in,2)
        if "(E)" in S_pattern:
            cv2.putText(img, "Color Error", (900,190),cv2.FONT_HERSHEY_SIMPLEX,1,text_color,5)
            cv2.putText(img, "Color Error", (900,190),cv2.FONT_HERSHEY_SIMPLEX,1,text_color_in,2)

def Store_face(Sort_pattern):
    global Out_FaceVerify_T
    global In_FaceVerify_T
    print('In Store_face')
    global Face
    global face_num 
    Face[face_num] = Sort_pattern
    face_num = face_num+1
    print(Face)
    print(face_num)
    Out_FaceVerify_T = In_FaceVerify_T


        
def Sign_BTW_Face(Srt_L , F_Step):
    #      1          
    #    5 3 2 6 
    #      4
    if face_num == 1:
        # ----->  3->2
        Arrow_Right(Srt_L)

    if face_num == 2:
        # ----->  2->6
        Arrow_Right(Srt_L)

    if face_num == 3:
        # ----->  6->5
        Arrow_Right(Srt_L)

    if face_num == 4:
        # ---> up 5->3->1
        if F_Step == 0:
            Arrow_Right(Srt_L)
            print('In Face 4 right')
        else:
            Arrow_Up(Srt_L)
            print('In Face 4 up')

    if face_num == 5:
        # down down 1->3->4
        Arrow_Down(Srt_L)
    if face_num == 6:
        Arrow_Up(Srt_L)

def Arrow_CW(Srt_L):
    cv2.arrowedLine(img, tuple(Srt_L[1]), tuple(Srt_L[5]), BG_color, 10)
    cv2.arrowedLine(img, tuple(Srt_L[5]), tuple(Srt_L[7]), BG_color, 10)
    cv2.arrowedLine(img, tuple(Srt_L[7]), tuple(Srt_L[3]), BG_color, 10)
    cv2.arrowedLine(img, tuple(Srt_L[3]), tuple(Srt_L[1]), BG_color, 10)
    cv2.arrowedLine(img, tuple(Srt_L[1]), tuple(Srt_L[5]), arrow_color, 7)
    cv2.arrowedLine(img, tuple(Srt_L[5]), tuple(Srt_L[7]), arrow_color, 7)
    cv2.arrowedLine(img, tuple(Srt_L[7]), tuple(Srt_L[3]), arrow_color, 7)
    cv2.arrowedLine(img, tuple(Srt_L[3]), tuple(Srt_L[1]), arrow_color, 7)
def Arrow_CCW(Srt_L):

    cv2.arrowedLine(img, tuple(Srt_L[5]), tuple(Srt_L[1]), BG_color, 10)
    cv2.arrowedLine(img, tuple(Srt_L[7]), tuple(Srt_L[5]), BG_color, 10)
    cv2.arrowedLine(img, tuple(Srt_L[3]), tuple(Srt_L[7]), BG_color, 10)
    cv2.arrowedLine(img, tuple(Srt_L[1]), tuple(Srt_L[3]), BG_color, 10)
    cv2.arrowedLine(img, tuple(Srt_L[5]), tuple(Srt_L[1]), arrow_color, 7)
    cv2.arrowedLine(img, tuple(Srt_L[7]), tuple(Srt_L[5]), arrow_color, 7)
    cv2.arrowedLine(img, tuple(Srt_L[3]), tuple(Srt_L[7]), arrow_color, 7)
    cv2.arrowedLine(img, tuple(Srt_L[1]), tuple(Srt_L[3]), arrow_color, 7)
def Arrow_Up(Srt_L):
    cv2.arrowedLine(img, tuple(Srt_L[6]), tuple(Srt_L[0]), BG_color, 10)
    cv2.arrowedLine(img, tuple(Srt_L[7]), tuple(Srt_L[1]), BG_color, 10)
    cv2.arrowedLine(img, tuple(Srt_L[8]), tuple(Srt_L[2]), BG_color, 10)
    cv2.arrowedLine(img, tuple(Srt_L[6]), tuple(Srt_L[0]), arrow_color, 8)
    cv2.arrowedLine(img, tuple(Srt_L[7]), tuple(Srt_L[1]), arrow_color, 8)
    cv2.arrowedLine(img, tuple(Srt_L[8]), tuple(Srt_L[2]), arrow_color, 8)
def Arrow_Down(Srt_L):
    cv2.arrowedLine(img, tuple(Srt_L[0]), tuple(Srt_L[6]), BG_color, 10)
    cv2.arrowedLine(img, tuple(Srt_L[1]), tuple(Srt_L[7]), BG_color, 10)
    cv2.arrowedLine(img, tuple(Srt_L[2]), tuple(Srt_L[8]), BG_color, 10)
    cv2.arrowedLine(img, tuple(Srt_L[0]), tuple(Srt_L[6]), arrow_color, 8)
    cv2.arrowedLine(img, tuple(Srt_L[1]), tuple(Srt_L[7]), arrow_color, 8)
    cv2.arrowedLine(img, tuple(Srt_L[2]), tuple(Srt_L[8]), arrow_color, 8)
def Arrow_Right(Srt_L):
    cv2.arrowedLine(img, tuple(Srt_L[0]), tuple(Srt_L[2]), BG_color, 10)
    cv2.arrowedLine(img, tuple(Srt_L[3]), tuple(Srt_L[5]), BG_color, 10)
    cv2.arrowedLine(img, tuple(Srt_L[6]), tuple(Srt_L[8]), BG_color, 10)
    cv2.arrowedLine(img, tuple(Srt_L[0]), tuple(Srt_L[2]), arrow_color, 8)
    cv2.arrowedLine(img, tuple(Srt_L[3]), tuple(Srt_L[5]), arrow_color, 8)
    cv2.arrowedLine(img, tuple(Srt_L[6]), tuple(Srt_L[8]), arrow_color, 8)
def Arrow_left(Srt_L):
    cv2.arrowedLine(img, tuple(Srt_L[2]), tuple(Srt_L[0]), BG_color, 10)
    cv2.arrowedLine(img, tuple(Srt_L[5]), tuple(Srt_L[3]), BG_color, 10)
    cv2.arrowedLine(img, tuple(Srt_L[8]), tuple(Srt_L[6]), BG_color, 10)
    cv2.arrowedLine(img, tuple(Srt_L[2]), tuple(Srt_L[0]), arrow_color, 8)
    cv2.arrowedLine(img, tuple(Srt_L[5]), tuple(Srt_L[3]), arrow_color, 8)
    cv2.arrowedLine(img, tuple(Srt_L[8]), tuple(Srt_L[6]), arrow_color, 8)

def Convert_face(SumFace):
    New_Sum = ''

    for x in SumFace:
        if x == SumFace[4]:
            New_Sum += 'U'
        if x == SumFace[13]:
            New_Sum += 'R'
        if x == SumFace[22]:
            New_Sum += 'F'
        if x == SumFace[31]:
            New_Sum += 'D'
        if x == SumFace[40]:
            New_Sum += 'L'
        if x == SumFace[49]:
            New_Sum += 'B'
    return New_Sum

def Sum_all_face():
    global SumFace
    Solve = ''
    New_Face = [Face[4],Face[1],Face[0],Face[5],Face[3],Face[2]]
    SumFace = convert(New_Face)
    print('SumFace :',SumFace)
    SumFace = Convert_face(SumFace)
    print('SumFace :',SumFace)
    cv2.putText(img, "All Face Have Recorded", (800,120),cv2.FONT_HERSHEY_SIMPLEX,1,text_color,5)
    cv2.putText(img, "All Face Have Recorded", (800,120),cv2.FONT_HERSHEY_SIMPLEX,1,text_color_in,2)

def Solve_rubik():
    Solve = kociemba.solve(SumFace)
    cv2.putText(img, 'Solve : '+Solve, (100,900),cv2.FONT_HERSHEY_SIMPLEX,1,text_color_in,2)
    return Solve

def Show_pending():
    Edge_OS = 40
    Show_Pending_detail(Edge_OS)
    #horizontal
    cv2.rectangle(img, pt1=(Edge_OS,Edge_OS+90), pt2=(Edge_OS+360,Edge_OS+180), color=(0,255,0), thickness=1)

    cv2.rectangle(img, pt1=(Edge_OS+90,Edge_OS+30), pt2=(Edge_OS+180,Edge_OS+60), color=(0,255,0), thickness=1)
    cv2.rectangle(img, pt1=(Edge_OS,Edge_OS+120), pt2=(Edge_OS+360,Edge_OS+150), color=(0,255,0), thickness=1)
    cv2.rectangle(img, pt1=(Edge_OS+90,Edge_OS+210), pt2=(Edge_OS+180,Edge_OS+240), color=(0,255,0), thickness=1)

    #vertical
    cv2.rectangle(img, pt1=(Edge_OS+90,Edge_OS), pt2=(Edge_OS+180,Edge_OS+270), color=(0,255,0), thickness=1)

    cv2.rectangle(img, pt1=(Edge_OS+30,Edge_OS+90), pt2=(Edge_OS+60,Edge_OS+180), color=(0,255,0), thickness=1)
    cv2.rectangle(img, pt1=(Edge_OS+120,Edge_OS), pt2=(Edge_OS+150,Edge_OS+270), color=(0,255,0), thickness=1)
    cv2.rectangle(img, pt1=(Edge_OS+210,Edge_OS+90), pt2=(Edge_OS+240,Edge_OS+180), color=(0,255,0), thickness=1)
    cv2.rectangle(img, pt1=(Edge_OS+270,Edge_OS+90), pt2=(Edge_OS+300,Edge_OS+180), color=(0,255,0), thickness=1)
    cv2.rectangle(img, pt1=(Edge_OS+330,Edge_OS+90), pt2=(Edge_OS+360,Edge_OS+180), color=(0,255,0), thickness=1)
    
    if face_num < 7:
        FaceToColor(face_num)

def FaceToColor(num):

    pos = (num*9)-9
    for i,x in enumerate(Face[num-1]):
        if x == 'B':
            Face_Color_Pattern[pos+i]= Blue
        if x == 'R':
            Face_Color_Pattern[pos+i]= Red
        if x == 'G':
            Face_Color_Pattern[pos+i]= Green
        if x == 'Y':
            Face_Color_Pattern[pos+i]= Yellow
        if x == 'O':
            Face_Color_Pattern[pos+i]= Orange
        if x == 'W':
            Face_Color_Pattern[pos+i]= White
        
def Show_Pending_detail(Edge_OS):
    # Pending_detail(Edge_OS+90,Edge_OS,0)
    # Pending_detail(Edge_OS+180,Edge_OS+90,9)
    # Pending_detail(Edge_OS+90,Edge_OS+90,18)
    # Pending_detail(Edge_OS+90,Edge_OS+180,27)
    # Pending_detail(Edge_OS,Edge_OS+90,36)
    # Pending_detail(Edge_OS+270,Edge_OS+90,45)

    Pending_detail(Edge_OS+90,Edge_OS+90,0)
    Pending_detail(Edge_OS+180,Edge_OS+90,9)
    Pending_detail(Edge_OS+270,Edge_OS+90,18)
    Pending_detail(Edge_OS,Edge_OS+90,27)
    Pending_detail(Edge_OS+90,Edge_OS,36)
    Pending_detail(Edge_OS+90,Edge_OS+180,45)

def Pending_detail(start_P_X,start_P_Y,num):
    
    cv2.rectangle(img, pt1=(start_P_X,start_P_Y), pt2=(start_P_X+30,start_P_Y+30), color = Face_Color_Pattern[num], thickness=-1)
    cv2.rectangle(img, pt1=(start_P_X+30,start_P_Y), pt2=(start_P_X+60,start_P_Y+30), color = Face_Color_Pattern[num+1], thickness=-1)
    cv2.rectangle(img, pt1=(start_P_X+60,start_P_Y), pt2=(start_P_X+90,start_P_Y+30), color = Face_Color_Pattern[num+2], thickness=-1)

    cv2.rectangle(img, pt1=(start_P_X,start_P_Y+30), pt2=(start_P_X+30,start_P_Y+60), color = Face_Color_Pattern[num+3], thickness=-1)
    cv2.rectangle(img, pt1=(start_P_X+30,start_P_Y+30), pt2=(start_P_X+60,start_P_Y+60), color = Face_Color_Pattern[num+4], thickness=-1)
    cv2.rectangle(img, pt1=(start_P_X+60,start_P_Y+30), pt2=(start_P_X+90,start_P_Y+60), color = Face_Color_Pattern[num+5], thickness=-1)

    cv2.rectangle(img, pt1=(start_P_X,start_P_Y+60), pt2=(start_P_X+30,start_P_Y+90), color = Face_Color_Pattern[num+6], thickness=-1)
    cv2.rectangle(img, pt1=(start_P_X+30,start_P_Y+60), pt2=(start_P_X+60,start_P_Y+90), color = Face_Color_Pattern[num+7], thickness=-1)
    cv2.rectangle(img, pt1=(start_P_X+60,start_P_Y+60), pt2=(start_P_X+90,start_P_Y+90), color = Face_Color_Pattern[num+8], thickness=-1)
    pass

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(3, 1920)
cap.set(4, 1080)

Verify_HSV()

while (cap.isOpened()):
    _,frame = cap.read()
    img = imutils.rotate(frame, 180)
    blurred_img = cv2.GaussianBlur(img, (21,21),0)
    hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

    # show_size(img)

    find_hsv_value()

    mask = cv2.inRange(hsv_img, lower_color, upper_color)
    # /////SET/////
    mask = cv2.bitwise_not(mask)   #for all color hide when finding HSV
    mask = cv2.erode(mask, Ero_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for index,contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if 40000>area>10000:
            # cv2.drawContours(img, contour,-1,(0,255,0),3) #check corner
            # print(area)
            M = cv2.moments(contour)
            approx = cv2.approxPolyDP(contour,0.001*cv2.arcLength(contour, True), True)
            cv2.drawContours(img, [approx],0,(102,0,102),4)
            cv2.drawContours(img, [approx],0,(0,255,255),2)

            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            
            point = [cx,cy]
            
            block_location.append(point)
            
            cv2.circle(img,(cx,cy),7,(0,0,0),-1)
            cv2.circle(img,(cx,cy),5,(0,255,255),-1)

            cv2.putText(img, "Centre",(cx-25,cy-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color,5)
            cv2.putText(img, "X = "+str(cx),(cx-30,cy+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color,5)
            cv2.putText(img, "Y = "+str(cy),(cx-30,cy+40),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color,5)

            cv2.putText(img, "Centre",(cx-25,cy-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color_in,2)
            cv2.putText(img, "X = "+str(cx),(cx-30,cy+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color_in,2)
            cv2.putText(img, "Y = "+str(cy),(cx-30,cy+40),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color_in,2)

    Srt_L = SortingPoint(block_location)
    block_location.clear()
    # print('Sorted point : ',Srt_L)
    pattern,Num_of_point = get_color(hsv_img, Srt_L)
    color_text(Srt_L, pattern)

    Show_pending()

    if face_num < 7 :
        Face_verify(Full_face_color, Num_of_point,Srt_L)
    else :
        Sum_all_face()
        Root = Solve_rubik()
    print(Root)
    
    # Show_image = img
    # Show_mask = mask
    # Show_HSV = hsv_img
    # Show_blurred = blurred_img
    Show_image = cv2.resize(img,(1280,720))
    Show_blurred = cv2.resize(blurred_img,(640,480))
    Show_mask = cv2.resize(mask,(640,480))
    Show_HSV = cv2.resize(hsv_img,(640,480))

    cv2.imshow("Original Image",Show_image)
    #cv2.imshow("Blurred Image",Show_blurred)
    cv2.imshow("Selection",Show_mask)
    cv2.imshow("HSV",Show_HSV)
    


    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()