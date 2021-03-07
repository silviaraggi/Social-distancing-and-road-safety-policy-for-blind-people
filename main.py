from Social_Distancing import social_distancing
from Traffic_Light_Detect import ROI_TrafficLights
import Sidewalks
import cv2
from threading import Thread


if __name__ == '__main__':

    cap = cv2.VideoCapture("annu4.mp4")
    while True:
        status, photo = cap.read()
        # photo = cv2.rotate(photo, cv2.ROTATE_90_CLOCKWISE)
        ret = cv2.resize(photo, (500, 750))
        D, photo = social_distancing.face_recognition(cap, photo)
        sound = Thread(target=social_distancing.play_sound(D))
        sound.start()
        photo = cv2.resize(photo, (500, 750))
        tot_image = ret  # immagine "normale" da sommare alle ROI
        roi = ROI_TrafficLights(80, 100)  # definisco la classe per i semafori (threshold, kernel)
        result3 = roi.display_roi(ret, [40, 40])  # identifico le ROI
        final_result, color = roi.detect_color(status, result3)  # identifico i colori solo all'interno delle ROI che ho identificato
        speech = Thread(target=ROI_TrafficLights.tts(color))
        speech.start()
        check = 0
        output = cv2.add(final_result, photo)
        if D == 0 and color == 0:
            marciapiede,  say= Sidewalks.run(ret)
            speech1 = Thread(target=Sidewalks.navigator(say))
            speech1.start()
            check = 1
            output = cv2.addWeighted(output,1,marciapiede,check, 0)
        cv2.imshow('RISULTATO', output)
        if cv2.waitKey(1) == 13:
            break
    cv2.destroyAllWindows()
