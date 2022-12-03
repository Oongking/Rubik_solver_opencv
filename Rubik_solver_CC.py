import kociemba
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import time


text_color = (0,0,0)
text_color_in = (0,255,255)

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
    # upper_color = np.array([u_h,u_s,u_v])\

class imgprocessing:

    lower_color = np.array([0,0,0]) #for all
    upper_color = np.array([179,255,91]) #for all
    Ero_kernel = np.ones((15, 15), np.uint8)
    kernel = np.ones((50, 50), np.uint8)
    BG_color = (0,0,0)
    arrow_color = (0,255,0)

    def process(self, img):
        self.img = imutils.rotate(img, 180)
        self.blurred_img = cv2.GaussianBlur(self.img, (21,21),0)
        self.hsv_img = cv2.cvtColor(self.blurred_img, cv2.COLOR_BGR2HSV)
        self.yuv_img = cv2.cvtColor(self.blurred_img, cv2.COLOR_BGR2YUV)

    def TofindMask(self):
        mask = cv2.inRange(self.hsv_img, self.lower_color, self.upper_color)
        mask = cv2.bitwise_not(mask)   #for all color hide when finding HSV
        mask = cv2.erode(mask, self.Ero_kernel, iterations=1)
        self.mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        self.target = cv2.bitwise_and(self.img,self.img, mask=self.mask)

    def Colortext(self,Points_location, pattern):
        for i,color in enumerate(pattern):
            cv2.putText(self.img, "Color = "+ pattern[i], (Points_location[i][0]-38,Points_location[i][1]-12),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color,5)

            cv2.putText(self.img, "Color = "+ pattern[i], (Points_location[i][0]-38,Points_location[i][1]-12),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color_in,2)
    
        Xpos = 1700
        ypos = 80
        cv2.putText(self.img, "B = Blue "    , (Xpos, ypos),cv2.FONT_HERSHEY_SIMPLEX,0.6,text_color,5)
        cv2.putText(self.img, "R = Red "     , (Xpos, ypos+20),cv2.FONT_HERSHEY_SIMPLEX,0.6,text_color,5)
        cv2.putText(self.img, "G = Green "   , (Xpos, ypos+40),cv2.FONT_HERSHEY_SIMPLEX,0.6,text_color,5)
        cv2.putText(self.img, "Y = Yellow "  , (Xpos, ypos+60),cv2.FONT_HERSHEY_SIMPLEX,0.6,text_color,5)
        cv2.putText(self.img, "O = Orange "  , (Xpos, ypos+80),cv2.FONT_HERSHEY_SIMPLEX,0.6,text_color,5)
        cv2.putText(self.img, "W = White "   , (Xpos, ypos+100),cv2.FONT_HERSHEY_SIMPLEX,0.6,text_color,5)

        cv2.putText(self.img, "B = Blue "    , (Xpos, ypos),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,128,0),2)
        cv2.putText(self.img, "R = Red "     , (Xpos, ypos+20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(51,51,255),2)
        cv2.putText(self.img, "G = Green "   , (Xpos, ypos+40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.putText(self.img, "Y = Yellow "  , (Xpos, ypos+60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        cv2.putText(self.img, "O = Orange "  , (Xpos, ypos+80),cv2.FONT_HERSHEY_SIMPLEX,0.6,(51,153,255),2)
        cv2.putText(self.img, "W = White "   , (Xpos, ypos+100),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

    def Gui(self, Face):
        Edge_OS = 40

        if Face[0].colorstored == 1:
            self.Pending_detail(Edge_OS+90,Edge_OS+90,Face[0].ReturnColor())
        if Face[1].colorstored == 1:
            self.Pending_detail(Edge_OS+180,Edge_OS+90,Face[1].ReturnColor())
        if Face[2].colorstored == 1:
            self.Pending_detail(Edge_OS+270,Edge_OS+90,Face[2].ReturnColor())
        if Face[3].colorstored == 1:
            self.Pending_detail(Edge_OS,Edge_OS+90,Face[3].ReturnColor())
        if Face[4].colorstored == 1:    
            self.Pending_detail(Edge_OS+90,Edge_OS,Face[4].ReturnColor())
        if Face[5].colorstored == 1:
            self.Pending_detail(Edge_OS+90,Edge_OS+180,Face[5].ReturnColor())

        # BG
        #horizontal
        cv2.rectangle(self.img, pt1=(Edge_OS,Edge_OS+90), pt2=(Edge_OS+360,Edge_OS+180), color=(0,153,76), thickness=2)

        cv2.rectangle(self.img, pt1=(Edge_OS+90,Edge_OS+30), pt2=(Edge_OS+180,Edge_OS+60), color=(0,153,76), thickness=2)
        cv2.rectangle(self.img, pt1=(Edge_OS,Edge_OS+120), pt2=(Edge_OS+360,Edge_OS+150), color=(0,153,76), thickness=2)
        cv2.rectangle(self.img, pt1=(Edge_OS+90,Edge_OS+210), pt2=(Edge_OS+180,Edge_OS+240), color=(0,153,76), thickness=2)
        #vertical
        cv2.rectangle(self.img, pt1=(Edge_OS+90,Edge_OS), pt2=(Edge_OS+180,Edge_OS+270), color=(0,153,76), thickness=2)

        cv2.rectangle(self.img, pt1=(Edge_OS+30,Edge_OS+90), pt2=(Edge_OS+60,Edge_OS+180), color=(0,153,76), thickness=2)
        cv2.rectangle(self.img, pt1=(Edge_OS+120,Edge_OS), pt2=(Edge_OS+150,Edge_OS+270), color=(0,153,76), thickness=2)
        cv2.rectangle(self.img, pt1=(Edge_OS+210,Edge_OS+90), pt2=(Edge_OS+240,Edge_OS+180), color=(0,153,76), thickness=2)
        cv2.rectangle(self.img, pt1=(Edge_OS+270,Edge_OS+90), pt2=(Edge_OS+300,Edge_OS+180), color=(0,153,76), thickness=2)
        cv2.rectangle(self.img, pt1=(Edge_OS+330,Edge_OS+90), pt2=(Edge_OS+360,Edge_OS+180), color=(0,153,76), thickness=2)
        
        # color
        #horizontal
        cv2.rectangle(self.img, pt1=(Edge_OS,Edge_OS+90), pt2=(Edge_OS+360,Edge_OS+180), color=(0,255,0), thickness=1)

        cv2.rectangle(self.img, pt1=(Edge_OS+90,Edge_OS+30), pt2=(Edge_OS+180,Edge_OS+60), color=(0,255,0), thickness=1)
        cv2.rectangle(self.img, pt1=(Edge_OS,Edge_OS+120), pt2=(Edge_OS+360,Edge_OS+150), color=(0,255,0), thickness=1)
        cv2.rectangle(self.img, pt1=(Edge_OS+90,Edge_OS+210), pt2=(Edge_OS+180,Edge_OS+240), color=(0,255,0), thickness=1)
        #vertical
        cv2.rectangle(self.img, pt1=(Edge_OS+90,Edge_OS), pt2=(Edge_OS+180,Edge_OS+270), color=(0,255,0), thickness=1)

        cv2.rectangle(self.img, pt1=(Edge_OS+30,Edge_OS+90), pt2=(Edge_OS+60,Edge_OS+180), color=(0,255,0), thickness=1)
        cv2.rectangle(self.img, pt1=(Edge_OS+120,Edge_OS), pt2=(Edge_OS+150,Edge_OS+270), color=(0,255,0), thickness=1)
        cv2.rectangle(self.img, pt1=(Edge_OS+210,Edge_OS+90), pt2=(Edge_OS+240,Edge_OS+180), color=(0,255,0), thickness=1)
        cv2.rectangle(self.img, pt1=(Edge_OS+270,Edge_OS+90), pt2=(Edge_OS+300,Edge_OS+180), color=(0,255,0), thickness=1)
        cv2.rectangle(self.img, pt1=(Edge_OS+330,Edge_OS+90), pt2=(Edge_OS+360,Edge_OS+180), color=(0,255,0), thickness=1)

    def Pending_detail(self,start_P_X,start_P_Y,colorV):
    
        cv2.rectangle(self.img, pt1=(start_P_X,start_P_Y), pt2=(start_P_X+30,start_P_Y+30), color = colorV[0], thickness=-1)
        cv2.rectangle(self.img, pt1=(start_P_X+30,start_P_Y), pt2=(start_P_X+60,start_P_Y+30), color = colorV[1], thickness=-1)
        cv2.rectangle(self.img, pt1=(start_P_X+60,start_P_Y), pt2=(start_P_X+90,start_P_Y+30), color = colorV[2], thickness=-1)

        cv2.rectangle(self.img, pt1=(start_P_X,start_P_Y+30), pt2=(start_P_X+30,start_P_Y+60), color = colorV[3], thickness=-1)
        cv2.rectangle(self.img, pt1=(start_P_X+30,start_P_Y+30), pt2=(start_P_X+60,start_P_Y+60), color = colorV[4], thickness=-1)
        cv2.rectangle(self.img, pt1=(start_P_X+60,start_P_Y+30), pt2=(start_P_X+90,start_P_Y+60), color = colorV[5], thickness=-1)

        cv2.rectangle(self.img, pt1=(start_P_X,start_P_Y+60), pt2=(start_P_X+30,start_P_Y+90), color = colorV[6], thickness=-1)
        cv2.rectangle(self.img, pt1=(start_P_X+30,start_P_Y+60), pt2=(start_P_X+60,start_P_Y+90), color = colorV[7], thickness=-1)
        cv2.rectangle(self.img, pt1=(start_P_X+60,start_P_Y+60), pt2=(start_P_X+90,start_P_Y+90), color = colorV[8], thickness=-1)

    def Arrow_CW(self,Srt_L):
        cv2.arrowedLine(self.img, tuple(Srt_L[1]), tuple(Srt_L[5]), self.BG_color, 10)
        cv2.arrowedLine(self.img, tuple(Srt_L[5]), tuple(Srt_L[7]), self.BG_color, 10)
        cv2.arrowedLine(self.img, tuple(Srt_L[7]), tuple(Srt_L[3]), self.BG_color, 10)
        cv2.arrowedLine(self.img, tuple(Srt_L[3]), tuple(Srt_L[1]), self.BG_color, 10)
        cv2.arrowedLine(self.img, tuple(Srt_L[1]), tuple(Srt_L[5]), self.arrow_color, 7)
        cv2.arrowedLine(self.img, tuple(Srt_L[5]), tuple(Srt_L[7]), self.arrow_color, 7)
        cv2.arrowedLine(self.img, tuple(Srt_L[7]), tuple(Srt_L[3]), self.arrow_color, 7)
        cv2.arrowedLine(self.img, tuple(Srt_L[3]), tuple(Srt_L[1]), self.arrow_color, 7)
    def Arrow_CCW(self,Srt_L):

        cv2.arrowedLine(self.img, tuple(Srt_L[5]), tuple(Srt_L[1]), self.BG_color, 10)
        cv2.arrowedLine(self.img, tuple(Srt_L[7]), tuple(Srt_L[5]), self.BG_color, 10)
        cv2.arrowedLine(self.img, tuple(Srt_L[3]), tuple(Srt_L[7]), self.BG_color, 10)
        cv2.arrowedLine(self.img, tuple(Srt_L[1]), tuple(Srt_L[3]), self.BG_color, 10)
        cv2.arrowedLine(self.img, tuple(Srt_L[5]), tuple(Srt_L[1]), self.arrow_color, 7)
        cv2.arrowedLine(self.img, tuple(Srt_L[7]), tuple(Srt_L[5]), self.arrow_color, 7)
        cv2.arrowedLine(self.img, tuple(Srt_L[3]), tuple(Srt_L[7]), self.arrow_color, 7)
        cv2.arrowedLine(self.img, tuple(Srt_L[1]), tuple(Srt_L[3]), self.arrow_color, 7)
    def Arrow_Up(self,Srt_L):
        cv2.arrowedLine(self.img, tuple(Srt_L[6]), tuple(Srt_L[0]), self.BG_color, 10)
        cv2.arrowedLine(self.img, tuple(Srt_L[7]), tuple(Srt_L[1]), self.BG_color, 10)
        cv2.arrowedLine(self.img, tuple(Srt_L[8]), tuple(Srt_L[2]), self.BG_color, 10)
        cv2.arrowedLine(self.img, tuple(Srt_L[6]), tuple(Srt_L[0]), self.arrow_color, 8)
        cv2.arrowedLine(self.img, tuple(Srt_L[7]), tuple(Srt_L[1]), self.arrow_color, 8)
        cv2.arrowedLine(self.img, tuple(Srt_L[8]), tuple(Srt_L[2]), self.arrow_color, 8)
    def Arrow_Down(self,Srt_L):
        cv2.arrowedLine(self.img, tuple(Srt_L[0]), tuple(Srt_L[6]), self.BG_color, 10)
        cv2.arrowedLine(self.img, tuple(Srt_L[1]), tuple(Srt_L[7]), self.BG_color, 10)
        cv2.arrowedLine(self.img, tuple(Srt_L[2]), tuple(Srt_L[8]), self.BG_color, 10)
        cv2.arrowedLine(self.img, tuple(Srt_L[0]), tuple(Srt_L[6]), self.arrow_color, 8)
        cv2.arrowedLine(self.img, tuple(Srt_L[1]), tuple(Srt_L[7]), self.arrow_color, 8)
        cv2.arrowedLine(self.img, tuple(Srt_L[2]), tuple(Srt_L[8]), self.arrow_color, 8)
    def Arrow_Right(self,Srt_L):
        cv2.arrowedLine(self.img, tuple(Srt_L[0]), tuple(Srt_L[2]), self.BG_color, 10)
        cv2.arrowedLine(self.img, tuple(Srt_L[3]), tuple(Srt_L[5]), self.BG_color, 10)
        cv2.arrowedLine(self.img, tuple(Srt_L[6]), tuple(Srt_L[8]), self.BG_color, 10)
        cv2.arrowedLine(self.img, tuple(Srt_L[0]), tuple(Srt_L[2]), self.arrow_color, 8)
        cv2.arrowedLine(self.img, tuple(Srt_L[3]), tuple(Srt_L[5]), self.arrow_color, 8)
        cv2.arrowedLine(self.img, tuple(Srt_L[6]), tuple(Srt_L[8]), self.arrow_color, 8)
    def Arrow_left(self,Srt_L):
        cv2.arrowedLine(self.img, tuple(Srt_L[2]), tuple(Srt_L[0]), self.BG_color, 10)
        cv2.arrowedLine(self.img, tuple(Srt_L[5]), tuple(Srt_L[3]), self.BG_color, 10)
        cv2.arrowedLine(self.img, tuple(Srt_L[8]), tuple(Srt_L[6]), self.BG_color, 10)
        cv2.arrowedLine(self.img, tuple(Srt_L[2]), tuple(Srt_L[0]), self.arrow_color, 8)
        cv2.arrowedLine(self.img, tuple(Srt_L[5]), tuple(Srt_L[3]), self.arrow_color, 8)
        cv2.arrowedLine(self.img, tuple(Srt_L[8]), tuple(Srt_L[6]), self.arrow_color, 8)
    
    def SameF(self):
        cv2.putText(self.img, "Same Face . . .", (900,120),cv2.FONT_HERSHEY_SIMPLEX,1,text_color,5)
        cv2.putText(self.img, "Same Face . . .", (900,120),cv2.FONT_HERSHEY_SIMPLEX,1,text_color_in,2)
    def Verifying(self):
        cv2.putText(self.img, "Verifying . . . Time : " + str(Rubik.Diff_FaceVerify_T), (800,120),cv2.FONT_HERSHEY_SIMPLEX,1,text_color,5)
        cv2.putText(self.img, "Verifying . . . Time : " + str(Rubik.Diff_FaceVerify_T), (800,120),cv2.FONT_HERSHEY_SIMPLEX,1,text_color_in,2)
    def Double(self):
        cv2.putText(self.img, "Double : " + str(Rubik.Diff_FaceVerify_T), (900,160),cv2.FONT_HERSHEY_SIMPLEX,1,text_color,5)
        cv2.putText(self.img, "Double : " + str(Rubik.Diff_FaceVerify_T), (900,160),cv2.FONT_HERSHEY_SIMPLEX,1,text_color_in,2)
    def Turning(self):
        cv2.putText(self.img, "Turning . . .", (900,120),cv2.FONT_HERSHEY_SIMPLEX,1,text_color,5)
        cv2.putText(self.img, "Turning . . .", (900,120),cv2.FONT_HERSHEY_SIMPLEX,1,text_color_in,2)
    def DoubleTurning(self):
        cv2.putText(self.img, "Double Turning . . .", (900,120),cv2.FONT_HERSHEY_SIMPLEX,1,text_color,5)
        cv2.putText(self.img, "Double Turning . . .", (900,120),cv2.FONT_HERSHEY_SIMPLEX,1,text_color_in,2)
    def FaceNotFull(self):
        cv2.putText(self.img, "Face not Full", (900,150),cv2.FONT_HERSHEY_SIMPLEX,1,text_color,5)
        cv2.putText(self.img, "Face not Full", (900,150),cv2.FONT_HERSHEY_SIMPLEX,1,text_color_in,2)
    def ColorError(self):
        cv2.putText(self.img, "Color Error", (900,190),cv2.FONT_HERSHEY_SIMPLEX,1,text_color,5)
        cv2.putText(self.img, "Color Error", (900,190),cv2.FONT_HERSHEY_SIMPLEX,1,text_color_in,2)
    def AllReceive(self):

        cv2.putText(self.img, "All Face Have Recorded", (800,120),cv2.FONT_HERSHEY_SIMPLEX,1,text_color,5)
        cv2.putText(self.img, "All Face Have Recorded", (800,120),cv2.FONT_HERSHEY_SIMPLEX,1,text_color_in,2)
    def Root(self,root):
        cv2.putText(self.img, 'Solve : '+ root, (100,900),cv2.FONT_HERSHEY_SIMPLEX,1,text_color_in,2)
    def ReturnMask(self):
        return self.mask
    def Complete(self):
        cv2.putText(self.img, "----- Complete -----", (800,120),cv2.FONT_HERSHEY_SIMPLEX,1,text_color,5)
        cv2.putText(self.img, "----- Complete -----", (800,120),cv2.FONT_HERSHEY_SIMPLEX,1,text_color_in,2)
    
    def SolvePattern(self,key,invert,Srt_L):
        if invert == 0:
            if key == 'R':
                cv2.arrowedLine(self.img, tuple(Srt_L[8]), tuple(Srt_L[2]), self.BG_color, 10)
                cv2.arrowedLine(self.img, tuple(Srt_L[8]), tuple(Srt_L[2]), self.arrow_color, 8)
            if key == 'L':
                cv2.arrowedLine(self.img, tuple(Srt_L[0]), tuple(Srt_L[6]), self.BG_color, 10)
                cv2.arrowedLine(self.img, tuple(Srt_L[0]), tuple(Srt_L[6]), self.arrow_color, 8)
            if key == 'B':
                if Rubik.ForB < 3:
                    self.Arrow_Up(Srt_L)
                    cv2.arrowedLine(self.img, tuple(Srt_L[2]), tuple(Srt_L[0]), self.BG_color, 10)
                    cv2.arrowedLine(self.img, tuple(Srt_L[2]), tuple(Srt_L[0]), self.arrow_color, 8)
                else:
                    self.Arrow_Down(Srt_L)
            if key == 'D':
                cv2.arrowedLine(self.img, tuple(Srt_L[6]), tuple(Srt_L[8]), self.BG_color, 10)
                cv2.arrowedLine(self.img, tuple(Srt_L[6]), tuple(Srt_L[8]), self.arrow_color, 8)
            if key == 'F':
                self.Arrow_CW(Srt_L)
            if key == 'U':
                cv2.arrowedLine(self.img, tuple(Srt_L[2]), tuple(Srt_L[0]), self.BG_color, 10)
                cv2.arrowedLine(self.img, tuple(Srt_L[2]), tuple(Srt_L[0]), self.arrow_color, 8)
        elif invert == 1:
            if key == 'R':
                cv2.arrowedLine(self.img, tuple(Srt_L[2]), tuple(Srt_L[8]), self.BG_color, 10)
                cv2.arrowedLine(self.img, tuple(Srt_L[2]), tuple(Srt_L[8]), self.arrow_color, 8)
            if key == 'L':
                cv2.arrowedLine(self.img, tuple(Srt_L[6]), tuple(Srt_L[0]), self.BG_color, 10)
                cv2.arrowedLine(self.img, tuple(Srt_L[6]), tuple(Srt_L[0]), self.arrow_color, 8)
            if key == 'B':
                if Rubik.ForB < 3:
                    self.Arrow_Up(Srt_L)
                    cv2.arrowedLine(self.img, tuple(Srt_L[0]), tuple(Srt_L[2]), self.BG_color, 10)
                    cv2.arrowedLine(self.img, tuple(Srt_L[0]), tuple(Srt_L[2]), self.arrow_color, 8)
                else:
                    self.Arrow_Down(Srt_L)
            if key == 'D':
                cv2.arrowedLine(self.img, tuple(Srt_L[8]), tuple(Srt_L[6]), self.BG_color, 10)
                cv2.arrowedLine(self.img, tuple(Srt_L[8]), tuple(Srt_L[6]), self.arrow_color, 8)
            if key == 'F':
                self.Arrow_CCW(Srt_L)
            if key == 'U':
                cv2.arrowedLine(self.img, tuple(Srt_L[0]), tuple(Srt_L[2]), self.BG_color, 10)
                cv2.arrowedLine(self.img, tuple(Srt_L[0]), tuple(Srt_L[2]), self.arrow_color, 8)

class Rubik:

    def __init__(self,num):
        self.num  = num
        self.Image1 = imgprocessing()
        self.Srt_L = []
        self.Face = [Face(1),Face(2),Face(3),Face(4),Face(5),Face(6)]
        self.Face_step = 0
        self.Toggle = 0
        self.face_num = 0
        self.In_FaceVerify_T = 0
        self.Out_FaceVerify_T = 0
        self.Diff_FaceVerify_T = 0
        self.double = 0
        self.setdouble = 0
        self.ForB = 0
        self.Face1 = Face(7)
        self.waitTime = 500
        pass
    
    def SortingPoint(self,Points):
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

        # print('Sum : ',Sum_point)
        return Sum_point

    def Run(self,img):
        self.Image1.process(frame)
        self.Image1.TofindMask()
        self.contour()
        self.PatternObyO,self.Num_Point_On_Screen = self.get_color(self.Image1.hsv_img,self.Srt_L)
        self.Image1.Colortext(self.Srt_L, self.PatternObyO)
        self.Image1.Gui(self.Face)
        if self.face_num <7:
            self.Face_verify(self.PatternObyO,self.Num_Point_On_Screen,self.Srt_L)
        elif self.face_num == 7:
            self.Sum_all_face()
            self.Root = self.Solve_rubik(self.SumFace)
            self.Image1.Root(self.Root)
            self.SolvePatternObO = SolvePatternObO(self.Root)
            self.Face1 = self.Face
            self.Turning_Face(self.SolvePatternObO.Root[self.SolvePatternObO.SolveStep])
            print('1 Face[0] : ',self.Face[0])
            self.face_num +=1
        else:
            self.Image1.Root(self.Root)
            self.Solving(self.SolvePatternObO,self.Num_Point_On_Screen,self.PatternObyO)
            


        self.Show()

    def Solving(self, SolvePatternObO,Num_Point_On_Screen,PatternObyO):
        invert = 0
        key = ''
        
        self.FullSolveStep = len(SolvePatternObO.Root)
        print(SolvePatternObO.Root)
        if Num_Point_On_Screen==9 and "(E)" not in PatternObyO:
            
            if SolvePatternObO.SolveStep < self.FullSolveStep:
                if 'B' not in SolvePatternObO.Root[SolvePatternObO.SolveStep]:
                    if PatternObyO == self.Face1[0].Face_Color_Pattern:
                        self.In_FaceVerify_T = int(round(time.time() * 1000))
                elif self.ForB == 0:
                    self.ForB = 1
            
            if self.ForB == 1:
                if PatternObyO[4] == self.Face1[4].Face_Color_Pattern[4]:
                    self.ForB = 2
            elif self.ForB == 2:
                if PatternObyO == self.Face1[4].Face_Color_Pattern:
                    self.ForB = 3
            elif self.ForB == 3:
                if PatternObyO == self.Face1[0].Face_Color_Pattern:
                    self.In_FaceVerify_T = int(round(time.time() * 1000))
            
            if SolvePatternObO.SolveStep < self.FullSolveStep:
                if "'" in SolvePatternObO.Root[SolvePatternObO.SolveStep]:
                    invert = 1
                
                if "2" in SolvePatternObO.Root[SolvePatternObO.SolveStep]:
                    self.double = 1
             
                key = SolvePatternObO.Root[SolvePatternObO.SolveStep].replace("'", '')
                key = key.replace("2",'')

                self.Diff_FaceVerify_T = self.In_FaceVerify_T - self.Out_FaceVerify_T
                
                if self.Diff_FaceVerify_T > self.waitTime :
                    self.Face1 = self.Face
                    SolvePatternObO.SolveStep += 1
                    if SolvePatternObO.SolveStep < self.FullSolveStep:
                        self.Turning_Face(SolvePatternObO.Root[SolvePatternObO.SolveStep])
                    self.double = 0
                    self.ForB = 0
                    self.Out_FaceVerify_T = self.In_FaceVerify_T
                    
                if self.double == 1 and PatternObyO != self.Face1[0].Face_Color_Pattern:
                    self.Image1.DoubleTurning()
                    self.Image1.SolvePattern(key,invert, self.Srt_L)
                    self.Out_FaceVerify_T = self.In_FaceVerify_T
                elif self.Diff_FaceVerify_T <= 0:
                    print('ForB : ',self.ForB)
                    self.Image1.Turning()
                    self.Image1.SolvePattern(key,invert, self.Srt_L)
                    self.Out_FaceVerify_T = self.In_FaceVerify_T
                else:
                    self.Image1.Verifying()
                
            else:
                self.Image1.Complete()
        
        else:
            self.Out_FaceVerify_T = int(round(time.time() * 1000))
            if Num_Point_On_Screen!=9:
                self.Image1.FaceNotFull()
            if "(E)" in PatternObyO:
                self.Image1.ColorError()     

    def Solve_rubik(self,SumFace):
        Solve = kociemba.solve(SumFace)
        
        return Solve

    def Face_verify(self,PatternObyO,Num_Point_On_Screen,Srt_L):
        
        preFace = self.Face[self.face_num-1].Midcolor()
        FirstFace = self.Face[0].Midcolor()
        
        self.In_FaceVerify_T = int(round(time.time() * 1000))

        if Num_Point_On_Screen==9 and "(E)" not in PatternObyO:
        
            print('In Face_Verify')
    
            if self.Face_step == 1:
                self.Sign_BTW_Face(Srt_L, self.Face_step)

                self.Image1.SameF()


            if PatternObyO[4] == preFace:
                self.Toggle = 1

                self.Image1.SameF()

                self.Sign_BTW_Face(Srt_L, self.Face_step)
            
            else:

                if self.face_num == 1 or self.face_num == 2 or self.face_num == 3:
                    if PatternObyO[4] != preFace:
                        self.Toggle = 0

                elif self.face_num == 4:
                    if PatternObyO[4] != preFace and PatternObyO[4] != FirstFace:
                        self.Toggle = 0
                        self.Face_step = 0
                    if PatternObyO[4] == FirstFace:
                        self.Face_step = 1

                elif self.face_num == 5:

                    if PatternObyO[4] != preFace and PatternObyO[4] != FirstFace:
                        self.Toggle = 0
                    if PatternObyO[4] == FirstFace or PatternObyO[4] == self.Face[1].Midcolor() or PatternObyO[4] == self.Face[2].Midcolor() or PatternObyO[4] == self.Face[3].Midcolor():
                        self.Face_step = 1
    
            if self.face_num == 6:
                    self.Sign_BTW_Face(Srt_L, self.Face_step) 

                    if PatternObyO == self.Face[0].Face_Color_Pattern:
                        self.face_num += 1 

            elif Num_Point_On_Screen==9 and "(E)" not in PatternObyO and self.Toggle == 0:
                self.Diff_FaceVerify_T = self.In_FaceVerify_T - self.Out_FaceVerify_T
                
                self.Image1.Verifying()

                self.Face_step = 0
                # print(Diff_FaceVerify_T)

                if self.Diff_FaceVerify_T > 500 :

                    self.Store_face(PatternObyO)

                    # print('Store_face : ',Diff_FaceVerify_T)
        
        
        else:
            self.Out_FaceVerify_T = self.In_FaceVerify_T
            if Num_Point_On_Screen!=9:
                self.Image1.FaceNotFull()
            if "(E)" in PatternObyO:
                self.Image1.ColorError()
                
    def Store_face(self,PatternObyO):
        print('In Store_face')
        print(PatternObyO)
        self.Face[self.face_num].addFaceColor(PatternObyO)
        self.face_num = self.face_num+1
        print(Face)
        print(self.face_num)
        self.Out_FaceVerify_T = self.In_FaceVerify_T

    def Sign_BTW_Face(self,Srt_L , F_Step):
        #      1          
        #    5 3 2 6 
        #      4
        if self.face_num == 1:
            # ----->  3->2
            self.Image1.Arrow_Right(Srt_L)

        if self.face_num == 2:
            # ----->  2->6
            self.Image1.Arrow_Right(Srt_L)

        if self.face_num == 3:
            # ----->  6->5
            self.Image1.Arrow_Right(Srt_L)

        if self.face_num == 4:
            # ---> up 5->3->1
            if F_Step == 0:
                self.Image1.Arrow_Right(Srt_L)
                print('In Face 4 right')
            else:
                self.Image1.Arrow_Up(Srt_L)
                print('In Face 4 up')

        if self.face_num == 5:
            # down down 1->3->4
            self.Image1.Arrow_Down(Srt_L)
        if self.face_num == 6:
            self.Image1.Arrow_Up(Srt_L)

    def contour(self):
        block_location = []
        contours = cv2.findContours(self.Image1.ReturnMask(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = imutils.grab_contours(contours)
        
        for index,contour in enumerate(self.contours):
            area = cv2.contourArea(contour)
            if 40000>area>10000:
                # cv2.drawContours(img, contour,-1,(0,255,0),3) #check corner
                # print(area)
                M = cv2.moments(contour)
                approx = cv2.approxPolyDP(contour,0.001*cv2.arcLength(contour, True), True)
                cv2.drawContours(self.Image1.img, [approx],0,(102,0,102),4)
                cv2.drawContours(self.Image1.img, [approx],0,(0,255,255),2)

                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
            
                point = [cx,cy]
            
                block_location.append(point)
            
                cv2.circle(self.Image1.img,(cx,cy),7,(0,0,0),-1)
                cv2.circle(self.Image1.img,(cx,cy),5,(0,255,255),-1)

                cv2.putText(self.Image1.img, "Centre",(cx-25,cy-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color,5)
                cv2.putText(self.Image1.img, "X = "+str(cx),(cx-30,cy+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color,5)
                cv2.putText(self.Image1.img, "Y = "+str(cy),(cx-30,cy+40),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color,5)

                cv2.putText(self.Image1.img, "Centre",(cx-25,cy-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color_in,2)
                cv2.putText(self.Image1.img, "X = "+str(cx),(cx-30,cy+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color_in,2)
                cv2.putText(self.Image1.img, "Y = "+str(cy),(cx-30,cy+40),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color_in,2)

        self.Srt_L = self.SortingPoint(block_location)
    
    def Show(self):
        Show_image = cv2.resize(self.Image1.img,(1280,720))
        Show_blurred = cv2.resize(self.Image1.blurred_img,(640,480))
        Show_mask = cv2.resize(self.Image1.mask,(640,480))
        Show_HSV = cv2.resize(self.Image1.hsv_img,(640,480))
        Show_YUV = cv2.resize(self.Image1.yuv_img,(640,480))
        Show_TARGET = cv2.resize(self.Image1.target,(640,480))


        cv2.imshow("Original Image",Show_image)
        #cv2.imshow("Blurred Image",Show_blurred)
        # cv2.imshow("Selection",Show_mask)
        cv2.imshow("HSV",Show_HSV)
        cv2.imshow("YUV",Show_YUV)
        cv2.imshow("TARGET",Show_TARGET)
        # print('YUV : ',self.Image1.yuv_img[self.Srt_L[0][1],self.Srt_L[0][0]])

    def convert(self,s):
        new = ""
        for x in s:
            new += x 
        return new

    def get_color(self,hsv_img, Srt_L):
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
        HSVofPoint = []
        colorpattern = []

        for i,_ in enumerate(Srt_L):
            HSVofPoint.append(hsv_img[Srt_L[i][1],Srt_L[i][0]])
        
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

       
        # self.Full_Face_color_line = self.convert(colorpattern)

        return colorpattern, len(colorpattern)        

    def Sum_all_face(self):

        Solve = ''
        New_Face = [self.Face[4].linePattern(),self.Face[1].linePattern(),self.Face[0].linePattern(),self.Face[5].linePattern(),self.Face[3].linePattern(),self.Face[2].linePattern()]
        SumFace = self.convert(New_Face)
        print('Color SumFace        :',SumFace)
        self.SumFace = self.Convert_face(SumFace)
        print('Coordinate SumFace   :',self.SumFace)
        
        self.Image1.AllReceive()
    
    def Convert_face(self,SumFace):
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

    def Turning_Face(self,SolvePatternObO):
        BufferFace = ['','','']
        if 'R' in SolvePatternObO:
            if "'" not in SolvePatternObO:
                if '2' in SolvePatternObO:
                    BufferFace = [self.Face[0].Face_Color_Pattern[2],self.Face[0].Face_Color_Pattern[5],self.Face[0].Face_Color_Pattern[8]]
                    for i in range(3):
                        self.Face[0].Face_Color_Pattern[2+(i*3)] = self.Face[5].Face_Color_Pattern[2+(i*3)]
                        self.Face[5].Face_Color_Pattern[2+(i*3)] = self.Face[2].Face_Color_Pattern[6-(i*3)]
                        self.Face[2].Face_Color_Pattern[6-(i*3)] = self.Face[4].Face_Color_Pattern[2+(i*3)]
                        self.Face[4].Face_Color_Pattern[2+(i*3)] = BufferFace[i]
                    self.Face[1].CWTwist()
                BufferFace = [self.Face[0].Face_Color_Pattern[2],self.Face[0].Face_Color_Pattern[5],self.Face[0].Face_Color_Pattern[8]]
                for i in range(3):
                    self.Face[0].Face_Color_Pattern[2+(i*3)] = self.Face[5].Face_Color_Pattern[2+(i*3)]
                    self.Face[5].Face_Color_Pattern[2+(i*3)] = self.Face[2].Face_Color_Pattern[6-(i*3)]
                    self.Face[2].Face_Color_Pattern[6-(i*3)] = self.Face[4].Face_Color_Pattern[2+(i*3)]
                    self.Face[4].Face_Color_Pattern[2+(i*3)] = BufferFace[i]
                self.Face[1].CWTwist()
            else:
                BufferFace = [self.Face[0].Face_Color_Pattern[2],self.Face[0].Face_Color_Pattern[5],self.Face[0].Face_Color_Pattern[8]]
                for i in range(3):
                    self.Face[0].Face_Color_Pattern[2+(i*3)] = self.Face[4].Face_Color_Pattern[2+(i*3)]
                    self.Face[4].Face_Color_Pattern[2+(i*3)] = self.Face[2].Face_Color_Pattern[6-(i*3)]
                    self.Face[2].Face_Color_Pattern[6-(i*3)] = self.Face[5].Face_Color_Pattern[2+(i*3)]
                    self.Face[5].Face_Color_Pattern[2+(i*3)] = BufferFace[i]
                self.Face[1].CCWTwist()

        if 'L' in SolvePatternObO:
            if "'" not in SolvePatternObO:
                if '2' in SolvePatternObO:
                    BufferFace = [self.Face[0].Face_Color_Pattern[0],self.Face[0].Face_Color_Pattern[3],self.Face[0].Face_Color_Pattern[6]]
                    for i in range(3):
                        self.Face[0].Face_Color_Pattern[i*3]        = self.Face[4].Face_Color_Pattern[i*3]
                        self.Face[4].Face_Color_Pattern[i*3]        = self.Face[2].Face_Color_Pattern[8-(i*3)]
                        self.Face[2].Face_Color_Pattern[8-(i*3)]    = self.Face[5].Face_Color_Pattern[i*3]
                        self.Face[5].Face_Color_Pattern[i*3]        = BufferFace[i]
                    self.Face[3].CWTwist()
                BufferFace = [self.Face[0].Face_Color_Pattern[0],self.Face[0].Face_Color_Pattern[3],self.Face[0].Face_Color_Pattern[6]]
                for i in range(3):
                    self.Face[0].Face_Color_Pattern[i*3]        = self.Face[4].Face_Color_Pattern[i*3]
                    self.Face[4].Face_Color_Pattern[i*3]        = self.Face[2].Face_Color_Pattern[8-(i*3)]
                    self.Face[2].Face_Color_Pattern[8-(i*3)]    = self.Face[5].Face_Color_Pattern[i*3]
                    self.Face[5].Face_Color_Pattern[i*3]        = BufferFace[i]
                self.Face[3].CWTwist()
            else:
                BufferFace = [self.Face[0].Face_Color_Pattern[0],self.Face[0].Face_Color_Pattern[3],self.Face[0].Face_Color_Pattern[6]]
                for i in range(3):
                    self.Face[0].Face_Color_Pattern[i*3]        = self.Face[5].Face_Color_Pattern[i*3]
                    self.Face[5].Face_Color_Pattern[i*3]        = self.Face[2].Face_Color_Pattern[8-(i*3)]
                    self.Face[2].Face_Color_Pattern[8-(i*3)]    = self.Face[4].Face_Color_Pattern[i*3]
                    self.Face[4].Face_Color_Pattern[i*3]        = BufferFace[i]
                self.Face[3].CCWTwist()
        
        if 'U' in SolvePatternObO:
            if "'" not in SolvePatternObO:
                if '2' in SolvePatternObO:
                    BufferFace = [self.Face[0].Face_Color_Pattern[0],self.Face[0].Face_Color_Pattern[1],self.Face[0].Face_Color_Pattern[2]]
                    for i in range(3):
                        self.Face[0].Face_Color_Pattern[i] = self.Face[1].Face_Color_Pattern[i]
                        self.Face[1].Face_Color_Pattern[i] = self.Face[2].Face_Color_Pattern[i]
                        self.Face[2].Face_Color_Pattern[i] = self.Face[3].Face_Color_Pattern[i]
                        self.Face[3].Face_Color_Pattern[i] = BufferFace[i]
                    self.Face[4].CWTwist()
                BufferFace = [self.Face[0].Face_Color_Pattern[0],self.Face[0].Face_Color_Pattern[1],self.Face[0].Face_Color_Pattern[2]]
                for i in range(3):
                    self.Face[0].Face_Color_Pattern[i] = self.Face[1].Face_Color_Pattern[i]
                    self.Face[1].Face_Color_Pattern[i] = self.Face[2].Face_Color_Pattern[i]
                    self.Face[2].Face_Color_Pattern[i] = self.Face[3].Face_Color_Pattern[i]
                    self.Face[3].Face_Color_Pattern[i] = BufferFace[i]
                self.Face[4].CWTwist()
            else:
                BufferFace = [self.Face[0].Face_Color_Pattern[0],self.Face[0].Face_Color_Pattern[1],self.Face[0].Face_Color_Pattern[2]]
                for i in range(3):
                    self.Face[0].Face_Color_Pattern[i] = self.Face[3].Face_Color_Pattern[i]
                    self.Face[3].Face_Color_Pattern[i] = self.Face[2].Face_Color_Pattern[i]
                    self.Face[2].Face_Color_Pattern[i] = self.Face[1].Face_Color_Pattern[i]
                    self.Face[1].Face_Color_Pattern[i] = BufferFace[i]
                self.Face[4].CCWTwist()
        
        if 'D' in SolvePatternObO:
            if "'" not in SolvePatternObO:
                if '2' in SolvePatternObO:
                    BufferFace = [self.Face[0].Face_Color_Pattern[6],self.Face[0].Face_Color_Pattern[7],self.Face[0].Face_Color_Pattern[8]]
                    for i in range(3):
                        self.Face[0].Face_Color_Pattern[6+i] = self.Face[3].Face_Color_Pattern[6+i]
                        self.Face[3].Face_Color_Pattern[6+i] = self.Face[2].Face_Color_Pattern[6+i]
                        self.Face[2].Face_Color_Pattern[6+i] = self.Face[1].Face_Color_Pattern[6+i]
                        self.Face[1].Face_Color_Pattern[6+i] = BufferFace[i]
                    self.Face[5].CWTwist()
                BufferFace = [self.Face[0].Face_Color_Pattern[6],self.Face[0].Face_Color_Pattern[7],self.Face[0].Face_Color_Pattern[8]]
                for i in range(3):
                    self.Face[0].Face_Color_Pattern[6+i] = self.Face[3].Face_Color_Pattern[6+i]
                    self.Face[3].Face_Color_Pattern[6+i] = self.Face[2].Face_Color_Pattern[6+i]
                    self.Face[2].Face_Color_Pattern[6+i] = self.Face[1].Face_Color_Pattern[6+i]
                    self.Face[1].Face_Color_Pattern[6+i] = BufferFace[i]
                self.Face[5].CWTwist()
            else:
                BufferFace = [self.Face[0].Face_Color_Pattern[6],self.Face[0].Face_Color_Pattern[7],self.Face[0].Face_Color_Pattern[8]]
                for i in range(3):
                    self.Face[0].Face_Color_Pattern[6+i] = self.Face[1].Face_Color_Pattern[6+i]
                    self.Face[1].Face_Color_Pattern[6+i] = self.Face[2].Face_Color_Pattern[6+i]
                    self.Face[2].Face_Color_Pattern[6+i] = self.Face[3].Face_Color_Pattern[6+i]
                    self.Face[3].Face_Color_Pattern[6+i] = BufferFace[i]
                self.Face[5].CCWTwist()

        if 'F' in SolvePatternObO:
            if "'" not in SolvePatternObO:
                if '2' in SolvePatternObO:
                    BufferFace = [self.Face[1].Face_Color_Pattern[0],self.Face[1].Face_Color_Pattern[3],self.Face[1].Face_Color_Pattern[6]]
                    for i in range(3):
                        self.Face[1].Face_Color_Pattern[i*3] = self.Face[4].Face_Color_Pattern[6+i]
                        self.Face[4].Face_Color_Pattern[6+i] = self.Face[3].Face_Color_Pattern[8-(i*3)]
                        self.Face[3].Face_Color_Pattern[8-(i*3)] = self.Face[5].Face_Color_Pattern[2-i]
                        self.Face[5].Face_Color_Pattern[2-i] = BufferFace[i]
                    self.Face[0].CWTwist()
                BufferFace = [self.Face[1].Face_Color_Pattern[0],self.Face[1].Face_Color_Pattern[3],self.Face[1].Face_Color_Pattern[6]]
                for i in range(3):
                    self.Face[1].Face_Color_Pattern[i*3] = self.Face[4].Face_Color_Pattern[6+i]
                    self.Face[4].Face_Color_Pattern[6+i] = self.Face[3].Face_Color_Pattern[8-(i*3)]
                    self.Face[3].Face_Color_Pattern[8-(i*3)] = self.Face[5].Face_Color_Pattern[2-i]
                    self.Face[5].Face_Color_Pattern[2-i] = BufferFace[i]
                self.Face[0].CWTwist()
            else:
                BufferFace = [self.Face[1].Face_Color_Pattern[0],self.Face[1].Face_Color_Pattern[3],self.Face[1].Face_Color_Pattern[6]]
                for i in range(3):
                    self.Face[1].Face_Color_Pattern[i*3] = self.Face[5].Face_Color_Pattern[2-i]
                    self.Face[5].Face_Color_Pattern[2-i] = self.Face[3].Face_Color_Pattern[8-(i*3)]
                    self.Face[3].Face_Color_Pattern[8-(i*3)] = self.Face[4].Face_Color_Pattern[6+i]
                    self.Face[4].Face_Color_Pattern[6+i] = BufferFace[i]
                self.Face[0].CCWTwist()

        if 'B' in SolvePatternObO:
            if "'" not in SolvePatternObO:
                if '2' in SolvePatternObO:
                    BufferFace = [self.Face[1].Face_Color_Pattern[2],self.Face[1].Face_Color_Pattern[5],self.Face[1].Face_Color_Pattern[8]]
                    for i in range(3):
                        self.Face[1].Face_Color_Pattern[2+(i*3)] = self.Face[5].Face_Color_Pattern[8-i]
                        self.Face[5].Face_Color_Pattern[8-i] = self.Face[3].Face_Color_Pattern[6-(i*3)]
                        self.Face[3].Face_Color_Pattern[6-(i*3)] = self.Face[4].Face_Color_Pattern[i]
                        self.Face[4].Face_Color_Pattern[i] = BufferFace[i]
                    self.Face[2].CWTwist()
                BufferFace = [self.Face[1].Face_Color_Pattern[2],self.Face[1].Face_Color_Pattern[5],self.Face[1].Face_Color_Pattern[8]]
                for i in range(3):
                    self.Face[1].Face_Color_Pattern[2+(i*3)] = self.Face[5].Face_Color_Pattern[8-i]
                    self.Face[5].Face_Color_Pattern[8-i] = self.Face[3].Face_Color_Pattern[6-(i*3)]
                    self.Face[3].Face_Color_Pattern[6-(i*3)] = self.Face[4].Face_Color_Pattern[i]
                    self.Face[4].Face_Color_Pattern[i] = BufferFace[i]
                self.Face[2].CWTwist()
            else:
                BufferFace = [self.Face[1].Face_Color_Pattern[2],self.Face[1].Face_Color_Pattern[5],self.Face[1].Face_Color_Pattern[8]]
                for i in range(3):
                    self.Face[1].Face_Color_Pattern[2+(i*3)] = self.Face[4].Face_Color_Pattern[i]
                    self.Face[4].Face_Color_Pattern[i] = self.Face[3].Face_Color_Pattern[6-(i*3)]
                    self.Face[3].Face_Color_Pattern[6-(i*3)] = self.Face[5].Face_Color_Pattern[8-i]
                    self.Face[5].Face_Color_Pattern[8-i] = BufferFace[i]
                self.Face[2].CCWTwist()

class Face:
    def __init__(self,num):
        self.num = num
        self.Face_Color_Pattern = ['' for x in range(9)]
        self.ColorValue = [(0,0,0) for x in range(9)]
        self.colorstored = 0
        
    def addFaceColor(self,Face_Color_Pattern):
        self.colorstored = 1
        self.Face_Color_Pattern = Face_Color_Pattern
        
    def ReturnColor(self):
        self.FaceToColor(self.Face_Color_Pattern)
        return self.ColorValue
    def Midcolor(self):
        return self.Face_Color_Pattern[4]
    def Topcolor(self):
        return self.Face_Color_Pattern[1]
    def Leftcolor(self):
        return self.Face_Color_Pattern[3]
    def Rightcolor(self):
        return self.Face_Color_Pattern[5]
    def Botcolor(self):
        return self.Face_Color_Pattern[7]

    def linePattern(self):
        return Rubik.convert(self.Face_Color_Pattern)

    def FaceToColor(self,Face_Color_Pattern):
        
        Blue    = [255,0,0]
        Red     = [0,0,255]
        Green   = [0,255,0]
        Yellow  = [0,255,255]
        Orange  = [0,128,255]
        White   = [255,255,255]
        for i,x in enumerate(Face_Color_Pattern):
            if x == 'B':
                self.ColorValue[i]= Blue
            if x == 'R':
                self.ColorValue[i]= Red
            if x == 'G':
                self.ColorValue[i]= Green
            if x == 'Y':
                self.ColorValue[i]= Yellow
            if x == 'O':
                self.ColorValue[i]= Orange
            if x == 'W':
                self.ColorValue[i]= White

    def CWTwist(self):
        self.Face_Color_Pattern = [self.Face_Color_Pattern[6],self.Face_Color_Pattern[3],self.Face_Color_Pattern[0],self.Face_Color_Pattern[7],self.Face_Color_Pattern[4],self.Face_Color_Pattern[1],self.Face_Color_Pattern[8],self.Face_Color_Pattern[5],self.Face_Color_Pattern[2]]
    def CCWTwist(self):
        self.Face_Color_Pattern = [self.Face_Color_Pattern[2],self.Face_Color_Pattern[5],self.Face_Color_Pattern[8],self.Face_Color_Pattern[1],self.Face_Color_Pattern[4],self.Face_Color_Pattern[7],self.Face_Color_Pattern[0],self.Face_Color_Pattern[3],self.Face_Color_Pattern[6]]
class SolvePatternObO():
    def __init__(self, Root):
        self.Root = Root.split(sep=' ')
        self.SolveStep = 0



cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_EXPOSURE, 0.001)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)
cap.set(3, 1920)
cap.set(4, 1080)

Rubik = Rubik(1)

while (cap.isOpened()):
    _,frame = cap.read()

    Rubik.Run(frame)
    
    print(Rubik.Srt_L)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()