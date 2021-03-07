import cv2
import numpy as np
import pyttsx3 as tts

class ROI_TrafficLights:  # classe per ottenere la ROI che segue i punti più luminosi del video
    number_of_frame_red = 0
    number_of_frame_green = 0
    number_of_frame_yellow = 0

    def tts(self):
        engine = tts.init()
        if self == 1:
            engine.say("Fermo")
        if self == 2:
            engine.say("Fermo")
        if self == 3:
            engine.say("Vai")
        engine.runAndWait()

    def __init__(self, threshold, kernel_size): #inizializzaione della classe
        self.threshold = threshold
        self.kernel = np.ones((kernel_size, kernel_size), dtype=int)
        self.params = cv2.SimpleBlobDetector_Params() #funzione per identificare Blob all'interno dell'immagine
        #parametri per la definizione dei Blob
        self.params.minThreshold = 50
        self.params.maxThreshold = 1000

        self.params.filterByArea = True
        self.params.minArea = 50
        self.params.maxArea = 1000

        self.params.filterByCircularity = True
        self.params.minCircularity = 0.05

        self.params.filterByConvexity = True
        self.params.maxConvexity = 1

        self.params.filterByInertia = True
        self.params.minInertiaRatio = 0.01

        self.params.filterByColor = False




    def compute_roi(self, video): #delimita la zona dove cercare possibili semafori

        frame_transformed_to_gray = cv2.cvtColor(video, cv2.COLOR_RGB2GRAY)#conversione in scala di grigi
        mask = np.zeros(video.shape[:2], dtype=np.uint8)
        mask = cv2.rectangle(mask, (175, 0), (475, 500), (255), thickness=-1)#delimito la ROI dove possono esserci semafori
        tophat = cv2.morphologyEx(frame_transformed_to_gray, cv2.MORPH_TOPHAT, self.kernel)#esegue la hat morphology per trovare le spot light
       # cv2.imshow("tophat", tophat)
        ret, thresh = cv2.threshold(tophat, self.threshold, 255, cv2.THRESH_BINARY)#segmentazione
       # cv2.imshow("threshold", thresh)


        dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        ret, markers = cv2.connectedComponents(np.uint8(dist_transform))
        watershed = cv2.watershed(video, markers)#watershed, o "spartiacque"
        watershed = cv2.normalize(watershed, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        #cv2.imshow("watershed", watershed)
        markers += 1

        ver = (cv2.__version__).split('.') #detect dei BLOBs
        if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(self.params)
        else:
            detector = cv2.SimpleBlobDetector_create(self.params)
        keypoints = detector.detect(watershed, mask) #la detect sulla watershed mi permette di eliminare il più possibile "falsi semafori"
        blobs = cv2.drawKeypoints(watershed, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #disegno i Blob trovati
      #  cv2.imshow('BLOB in watershed', blobs)

        for keypoint in keypoints:#per ogni keypoint, ovvero per ogni Blob (possibile semaforo trovato) ne prendo le coordinate del centro
            x = keypoint.pt[0]
            y = keypoint.pt[1]
            coordinate_x = int(np.median(x))
            coordinate_y = int(np.median(y))
            yield coordinate_x, coordinate_y

    def display_roi(self, video, window_size):#costruisco finestre attorno ai possibili semafori, definizione delle ROI

        mask = np.zeros(video.shape, dtype=np.uint8)
        x_offset = int((window_size[0] - 1) / 2)
        y_offset = int((window_size[1] - 1) / 2)
        display_img = np.zeros(video.shape, dtype=np.uint8)

        for (x, y) in self.compute_roi(video):#richiamo la compute_roi da dove ottengo le coordinate dei blob e costruisco così le ROI
            x_min, x_max = x - x_offset, x + x_offset
            y_min, y_max = y - y_offset, y + y_offset
            mask[y_min:y_max, x_min:x_max] = 1
            display_img = np.zeros(video.shape, dtype=np.uint8)
            display_img[mask == 1] = video[mask == 1]

        return display_img

    def detect_color(self, ret, img): #rileva i colori dei semafori attraverso l'uso delle maschere di colore
        font = cv2.FONT_HERSHEY_SIMPLEX
        cimg = img
        hsv = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)#normalizzo l'immagine per togliere la luce
        color=0

        if ret is True:
            hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV) #prima bisogna convertire l'immagine in HSV
            lower_red1 = np.array([0, 100, 100]) #parametri dei colori in HSV
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            lower_green1 = np.array([40, 50, 50])
            upper_green1 = np.array([90, 260, 260])
            lower_yellow = np.array([15, 150, 150])
            upper_yellow = np.array([35, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1) #costruzione delle maschere di colore (binco/nero)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
            maskg = cv2.inRange(hsv, lower_green1, upper_green1)
            maskr = cv2.add(mask1, mask2)
            #definizione delle maschere con colori
            red_Mask = cv2.bitwise_and(cimg, cimg, mask=maskr)
            yellow_Mask = cv2.bitwise_and(cimg, cimg, mask=masky)
            green_Mask = cv2.bitwise_and(cimg, cimg, mask=maskg)

            if maskr is not None: #MASCHERA ROSSA

                ver = (cv2.__version__).split('.')
                if int(ver[0]) < 3:
                    detector = cv2.SimpleBlobDetector(self.params)
                else:
                    detector = cv2.SimpleBlobDetector_create(self.params)
                keypoints_r = detector.detect(red_Mask) #RILEVA BLOB ROSSI

                for keypoint in keypoints_r: #PER OGNI BLOB ROSSO SEGNA UN CERCHIO E LA SCRITTA
                    x = keypoint.pt[0]
                    y = keypoint.pt[1]
                    coordinate_x = int(np.median(x))
                    coordinate_y = int(np.median(y))

                    cv2.putText(cimg, 'RED', (coordinate_x, coordinate_y), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.drawKeypoints(cimg, keypoints_r, np.array([]), (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    red_Mask = cv2.drawKeypoints(red_Mask, keypoints_r, np.array([]), (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    cv2.circle(cimg, (coordinate_x, coordinate_y), 10, (0, 255, 0), 2)
                    cv2.circle(red_Mask, (coordinate_x, coordinate_y), 5, (0, 255, 0), 2)
                    cv2.imshow("RED MASK", red_Mask)
                    ROI_TrafficLights.number_of_frame_red=ROI_TrafficLights.number_of_frame_red+1

                    if(ROI_TrafficLights.number_of_frame_red==3):
                        #cv2.putText(cimg, "suono_per_rosso", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 0, 255], 4)
                        #funzione per far dire al comando vocale è rosso
                        color = 1
                        ROI_TrafficLights.number_of_frame_yellow=0
                        ROI_TrafficLights.number_of_frame_green=0


            if maskg is not None: #MASCHERA VERDE
                ver = (cv2.__version__).split('.')
                if int(ver[0]) < 3:
                    detector = cv2.SimpleBlobDetector(self.params)
                else:
                    detector = cv2.SimpleBlobDetector_create(self.params)
                keypoints_g = detector.detect(green_Mask)

                for keypoint in keypoints_g:
                    x = keypoint.pt[0]
                    y = keypoint.pt[1]
                    # s=keypoint.size #diametro blob
                    coordinate_x = int(np.median(x))
                    coordinate_y = int(np.median(y))
                    cv2.putText(cimg, 'GREEN', (coordinate_x, coordinate_y), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.drawKeypoints(cimg, keypoints_g, np.array([]), (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    green_Mask = cv2.drawKeypoints(green_Mask, keypoints_g, np.array([]), (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    cv2.circle(cimg, (coordinate_x, coordinate_y), 10, (0, 255, 0), 2)
                    cv2.circle(green_Mask, (coordinate_x, coordinate_y), 5, (0, 255, 0), 2)
                   # cv2.imshow("GREEN MASK", green_Mask)
                    ROI_TrafficLights.number_of_frame_green = ROI_TrafficLights.number_of_frame_green + 1

                    if(ROI_TrafficLights.number_of_frame_green==3):
                    # funzione per far dire al comando vocale è verde
                    # cv2.putText(cimg, "suono_per_verde", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 0, 255], 4)
                        color = 3
                        ROI_TrafficLights.number_of_frame_yellow=0
                        ROI_TrafficLights.number_of_frame_red=0
                        # print(ROI_TrafficLights.number_of_frame_yellow)


            if masky is not None: #MASCHERA GIALLA
                ver = (cv2.__version__).split('.')
                if int(ver[0]) < 3:
                    detector = cv2.SimpleBlobDetector(self.params)
                else:
                    detector = cv2.SimpleBlobDetector_create(self.params)
                keypoints_y = detector.detect(yellow_Mask)

                for keypoint in keypoints_y:
                    x = keypoint.pt[0]
                    y = keypoint.pt[1]
                    coordinate_x = int(np.median(x))
                    coordinate_y = int(np.median(y))

                    cv2.putText(cimg, 'YELLOW', (coordinate_x, coordinate_y), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.drawKeypoints(cimg, keypoints_y, np.array([]), (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    yellow_Mask = cv2.drawKeypoints(yellow_Mask, keypoints_y, np.array([]), (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    cv2.circle(cimg, (coordinate_x, coordinate_y), 10, (0, 255, 0), 2)
                    cv2.circle(yellow_Mask, (coordinate_x, coordinate_y), 5, (0, 255, 0), 2)
                  #  cv2.imshow("YELLOW MASK", yellow_Mask)
                    ROI_TrafficLights.number_of_frame_yellow = ROI_TrafficLights.number_of_frame_yellow + 1
                    # print(ROI_TrafficLights.number_of_frame_yellow)

                    if(ROI_TrafficLights.number_of_frame_yellow==3):
                    #funzione per far dire al comando vocale è giallo
                    #   cv2.putText(cimg, "suono_per_giallo", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 0, 255], 4)
                        color = 2
                        ROI_TrafficLights.number_of_frame_green=0
                        ROI_TrafficLights.number_of_frame_red=0


            if keypoints_r is None and keypoints_y is None and keypoints_g is None or len(keypoints_r)+len(keypoints_g)+len(keypoints_y) ==0 :
                color = 0
        return cimg, color