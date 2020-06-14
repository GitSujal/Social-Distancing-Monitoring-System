import cv2 as cv

from scipy.spatial import distance as dist
from imutils import perspective
import imutils
import numpy as np




def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

ix,iy = -1,-1
# mouse callback function
def getMousePointer(event,x,y,flags,param):
    global ix,iy
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img,(x,y),3,(255,0,0),-1)
        ix,iy = x,y

img = cv.imread("refPicture1.png")

# Window name in which image is displayed 
win_name = 'Image'
  
refObj = []

pts_src = np.array([[561,1022],[990,698],[486,273],[95,504]],dtype='float32')
pts_dest = np.array([[0,0],[0,400],[400,700],[0,700]],dtype='float32')
# calculate matrix H
h, status = cv.findHomography(pts_src,pts_dest)

def distance(pt1,pt2):
    return int(((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**0.5)

def midpoint(pt1,pt2):
    return (int((pt1[0]+pt2[0])/2),int((pt1[1]+pt2[1])/2))


def actual_distance(pt1,pt2):
    # provide a point you wish to map from image 1 to image 2
    a1 = np.array([pt1], dtype='float32')
    a1 = np.array([a1])
    out1 = cv.perspectiveTransform(a1,h)
    # print(a1[0][0])
    # print(out1[0][0])
    
    a2 = np.array([pt2], dtype='float32')
    a2 = np.array([a2])
    out2 = cv.perspectiveTransform(a2,h)
    # print(a2[0][0])
    # print(out2[0][0])
    
    
    act_dist = distance(out1[0][0],out2[0][0])
    return act_dist


print(status)

# finally, get the mapping
#pointsOut = cv.perspectiveTransform(a, h)

while True:
    cv.imshow(win_name,img)
    cv.setMouseCallback(win_name, getMousePointer)
    if ix!=-1 and iy!=-1:
        refObj.append((ix,iy))
        ix=-1
        iy=-1
        # print(len(refObj))
    if len(refObj)==2:
        img = cv.imread("refPicture1.png") 
        # img = cv.warpPerspective(img, h, (1000,800))
        
        print("Len 2 Rectangle Drawn.")
        # print(refObj)
        img = cv.line(img,refObj[0],refObj[1],(255,0,0),2)
        dist = distance(refObj[0],refObj[1])
        
        cv.putText(img,str(dist),midpoint(refObj[0],refObj[1]),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),thickness=3)
        
        # # provide a point you wish to map from image 1 to image 2
        # a1 = np.array([refObj[0]], dtype='float32')
        # a1 = np.array([a1])
        # out1 = cv.perspectiveTransform(a1,h)
        # print(a1[0][0])
        # print(out1[0][0])
        
        # a2 = np.array([refObj[1]], dtype='float32')
        # a2 = np.array([a2])
        # out2 = cv.perspectiveTransform(a2,h)
        # print(a2[0][0])
        # print(out2[0][0])
        
        
        
        act_dist = actual_distance(refObj[0],refObj[1])
        x,y = midpoint(refObj[0],refObj[1])
        x=x
        y=y+50
        
        cv.putText(img,str(act_dist),(x,y),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),thickness=3)
        
        
        refObj.clear()
        
    # if len(refObj)==4:
    #     img = cv.imread("refPicture1.png")
    #     print(refObj)
    #     img = cv.line(img,refObj[0],refObj[1],(255,0,0),3)
    #     img = cv.line(img,refObj[1],refObj[2],(255,0,0),3)
    #     img = cv.line(img,refObj[2],refObj[3],(255,0,0),3)
    #     img = cv.line(img,refObj[3],refObj[0],(255,0,0),3)
    #     refObj.clear()
        
    if cv.waitKey(1) & 0xFF == ord('q'):
        break



