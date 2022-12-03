import kociemba
import cv2 as cv2
import numpy as np
import imutils

#              |************|   U = Yellow      
#              |*U1**U2**U3*|   L = Orange
#              |************|   F = Blue
#              |*U4**U5**U6*|   R = Red
#              |************|   B = Green
#              |*U7**U8**U9*|   D = White
#              |************|
#  ************|************|************|************
#  *L1**L2**L3*|*F1**F2**F3*|*R1**R2**R3*|*B1**B2**B3*
#  ************|************|************|************
#  *L4**L5**L6*|*F4**F5**F6*|*R4**R5**R6*|*B4**B5**B6*
#  ************|************|************|************
#  *L7**L8**L9*|*F7**F8**F9*|*R7**R8**R9*|*B7**B8**B9*
#  ************|************|************|************
#              |************|
#              |*D1**D2**D3*|
#              |************|
#              |*D4**D5**D6*|
#              |************|
#              |*D7**D8**D9*|
#              |************|

# U1, U2, U3, U4, U5, U6, U7, U8, U9, R1, R2, R3, R4, R5, R6, R7, R8, R9, F1, F2, F3, F4, F5, F6, F7, F8, F9, D1, D2, D3, D4, D5, D6, D7, D8, D9, L1, L2, L3, L4, L5, L6, L7, L8, L9, B1, B2, B3, B4, B5, B6, B7, B8, B9
# UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB

#https://github.com/muodov/kociemba
#https://www.hackster.io/hbolanos2001/rubik-s-cube-solver-robot-abbc9d

cap = cv2.VideoCapture(0)



while (cap.isOpened()):

    _,frame = cap.read()
    blurred_frame = cv2.GaussianBlur(frame, (15,15),0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    cv2.imshow("frame",frame)

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()