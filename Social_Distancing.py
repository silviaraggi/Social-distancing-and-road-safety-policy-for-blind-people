import cv2
from scipy.spatial import distance as dist
import math
from threading import Thread
import winsound
import numpy as np

class social_distancing:

    frequency = 3000  # Set Frequency To 2500 Hertz
    maxfaces = 5
    facenum = np.zeros(maxfaces)
    facecheck = 0
    faceflag=np.zeros(maxfaces)
    D = 0
    D_1=0

    def play_sound(self):
        D_new = int(social_distancing.D*math.exp(1.3))
        if self<150 and self!=0:
            winsound.Beep(social_distancing.frequency, D_new)


    def face_recognition(self, photo):

        face_model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        sensor_height = 3.6
        reference_face_height = 239
        reference_focal_length = 2.8
        #valori per iphone 8
        width = int(self.get(3))
        height = int(self.get(4))
        mask = np.zeros_like(photo)
        vert = np.array([[width/8, height*3/4], [width/8, height/6], [width*7/8, height/6], [width*7/8, height*3/4]], np.int32)
        cv2.fillPoly(mask, [vert], (255,255,255))
        roi = cv2.bitwise_and(photo, mask)
        # cv2.imshow('ROI', roi)
        face_cor = face_model.detectMultiScale(roi)
        output = np.zeros_like(photo)

        l = min(social_distancing.maxfaces-1,len(face_cor))

        if isinstance(social_distancing.facecheck,list) is False:
            social_distancing.facecheck = face_cor

        if l == 0:
            social_distancing.D = 0
            for i in range(0,social_distancing.maxfaces):
                social_distancing.facenum[i] = 0
            pass
        else:
            for i in range(0,l):
                x1 = face_cor[i][0]
                y1 = face_cor[i][1]
                x2 = face_cor[i][0] + face_cor[i][2]
                y2 = face_cor[i][1] + face_cor[i][3]
                face_height = y2-y1
                mid_x = int((x1+x2)/2)
                mid_y = int((y1+y2)/2)
                social_distancing.D1 = ((reference_focal_length * reference_face_height * height) / (face_height * sensor_height)) / 10
                if social_distancing.D1<300:
                    for j in range(0,l):
                        if social_distancing.facenum[j] != 0:
                            x1_j = social_distancing.facecheck[j][0]
                            y1_j = social_distancing.facecheck[j][1]
                            x2_j = social_distancing.facecheck[j][0] + social_distancing.facecheck[j][2]
                            y2_j = social_distancing.facecheck[j][1] + social_distancing.facecheck[j][3]
                            if x1 > x1_j - 10 and y1 > y1_j -10 and x2 < x2_j + 10 and y2 < y2_j + 10:
                                social_distancing.facenum[j] = social_distancing.facenum[j] + 1  # la faccia è nel range di quella precedente
                                social_distancing.faceflag[j]=1 #flag per evitare falsi positivi
                        else:
                            social_distancing.facenum[i] = social_distancing.facenum[i]+1
                if social_distancing.facenum[i] > 5:
                    social_distancing.D = ((reference_focal_length * reference_face_height * height)/(face_height*sensor_height))/10
                    if social_distancing.D < 300:
                        output = cv2.circle(output, (mid_x, mid_y), 3 , [255,0,0] , -1)
                        output = cv2.rectangle(output , (x1, y1) , (x2,y2) , [0,255,0] , 2)
                        output = cv2.putText(output, str(round(social_distancing.D,2)) + " cm", (x1, y2+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        if social_distancing.D<150 and social_distancing.D!=0:
                            output = cv2.putText(output, "Attenzione!!", (100, 400), cv2.FONT_HERSHEY_SIMPLEX,2, [0,0,255] , 4)
                    else:
                        social_distancing.D = 0
                        social_distancing.facenum[i] = 0
            for i in range(0,l):
                if social_distancing.faceflag[i]==0:
                    social_distancing.facenum[i]=0
                social_distancing.faceflag[i]=0
        social_distancing.facecheck = face_cor
        photo = cv2.add(output, photo)
        return social_distancing.D, photo