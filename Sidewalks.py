import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import pyttsx3 as tts

stop=0
straight=0
turn_left=0
turn_right=0

def roi(img, vert): #estraimo una region of interest
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vert, 255)
    return cv2.bitwise_and(img, mask)

def edgeDetect(img): #rilevamento bordi
    img = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(img, 100, 150)
    return edges

def navigator(self): #output sonoro
    engine = tts.init()
    if self == 1:
        engine.say("Continua dritto")
    if self == 2:
        engine.say("Ruota a destra")
    if self == 3:
        engine.say("Ruota a sinistra")
    if self == 4:
        engine.say("Stop")
    engine.runAndWait()

def run(screen):

    global stop
    global straight
    global turn_left
    global turn_right
    flagstop=1
    vert = np.array([[0, 725], [0, 500], [500, 500], [500, 725]], np.int32) #vertici region of interest
    fin = edgeDetect(screen)
    # cv2.imshow('edgeDetect', fin)
    fin = roi(fin, [vert])
    # cv2.imshow('ROI', fin)
    line = cv2.HoughLinesP(fin, 1, np.pi / 180, 10,None,50,10)
    output = np.zeros_like(screen)
    leftList=[]
    rightList=[]
    horizontal=[]
    cor_x = []
    cor_y = []
    say=0

    if line is None:
        line=[]
    if (len(line)!=0):
        for i in line:
            distance=math.sqrt(math.pow(i[0][0]-i[0][2],2)+math.pow(i[0][1]-i[0][3],2)) #calcolo lunghezza linea
            if i[0][1]<i[0][3]: #imponiamo p1 come il punto più in alto e p2 più basso così da avere la stessa notazione per tutto il codice
                p1=[i[0][0],i[0][1]]
                p2 = [i[0][2], i[0][3]]
            else:
                p1 = [i[0][2], i[0][3]]
                p2 = [i[0][0], i[0][1]]

            slope=math.atan2(p2[1]-p1[1], p2[0]-p1[0]) #calcolo angolo tra la linea e l'asse orizzontale
            if p2[1]-p1[1] !=0:
                m=(p2[0]-p1[0])/(p2[1]-p1[1]) #calcolo coefficiente angolare linea
            else:
                m=-np.inf

            #divisione in liste delle linee in base a angolo e posizione del punto più in basso
            if(p2[0]<=250 and slope > math.radians(95) and  slope< math.radians(150)):
                leftList.append([distance,p1,p2,slope,m])

            elif (p2[0]>250 and slope > math.radians(30) and  slope< math.radians(85)):
                rightList.append([distance,p1,p2,slope,m])

            elif(((slope > math.radians(0) and slope<math.radians(15)) or (slope > math.radians(165) and slope<math.radians(180))) and distance>200):
                horizontal.append([distance, p1, p2, slope, m])
        #ordinamento seguendo la lunghezza delle linee
        leftList.sort(reverse=True)
        rightList.sort(reverse=True)
        horizontal.sort(reverse=True)
        #consideriamo solo le linee più lunghe
        leftList=leftList[0:min(len(leftList), 2)]
        rightList = rightList[0:min(len(rightList), 2)]
        horizontal = horizontal[0:min(len(rightList), 2)]
        #stampa linee
        for i in leftList:
            cv2.line(output, (i[1][0], i[1][1]), (i[2][0], i[2][1]), (255, 0, 0), 10)

        for i in rightList:
            cv2.line(output, (i[1][0], i[1][1]), (i[2][0], i[2][1]), (255, 0, 0), 10)

        for i in horizontal:
            cv2.line(output, (i[1][0], i[1][1]), (i[2][0], i[2][1]), (255, 0, 0), 10)

        #nelle seguenti linee di codice si esegue il prolungamento delle linee in modo tale da valutare la loro intersezione

        for i in leftList:
            for j in rightList:
                y=min(i[2][1],j[2][1])
                flag=0
                x1=(i[2][1]-y)*-i[4]+i[2][0]
                x2 = ( j[2][1]- y ) * -j[4]+j[2][0]
                if (x1>x2):
                    flag=1
                while (y>0 and flag==0):
                    y=y-1
                    x1=x1-i[4]
                    x2=x2-j[4]
                    if (x1 > x2):
                        flag = 1
                cor_x.append((x1+x2)/2)
                cor_y.append(y)
        #si valuta l'intersezione ottenuta così da definire il comportamento da eseguire
        if not (len(cor_x)==0):
            intersection_x=sum(cor_x)/len(cor_x)
            intersection_y = sum(cor_y) / len(cor_y)
            cv2.circle(output, (int(intersection_x),int(intersection_y)), 10, [0, 0, 255], -5)
            if intersection_x<200:
                turn_left=0
                turn_right+=1
                straight=0
            elif intersection_x>300:
                turn_right=0
                turn_left+=1
                straight = 0
            else:
                straight+=1
                turn_left=0
                turn_right=0
        #si valuta se vi sono delle linee orizzontali appartenenti alla fine del marciapiede
        if (len(horizontal)!=0 and len(leftList)!=0 and len(rightList)!=0):
            for i in horizontal:
                for j in leftList:
                    for z in rightList:
                        y=max(i[1][1],i[2][1])
                        if(y>j[1][1] or y>z[1][1] or y>j[2][1] or y>z[2][1]):
                            flagstop=0

            if(flagstop==0):
                stop=0
            else:
                stop+=1
        else:
            stop=0

        if (straight == 10 and len(cor_x)!=0 ):
            say=1
        if (turn_left == 10 and len(cor_x)!=0):
            say=2
        if (turn_right == 10 and len(cor_x)!=0):
            say=3
        if(stop==10):
            say=4

    return output,say