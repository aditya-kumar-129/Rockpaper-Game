#-----------------------GENERAL IMPORTS----------------------
import random
from keras import models
import cv2
import cvzone
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # USED TO RUN ON CPU

#IMPORTING MODEL
model = models.load_model(r'C:\Users\adity\Desktop\mymodel.h5')

#SETTING GENERAL WINDOW SETTINGS
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # 640 is for width
cap.set(4, 480)  # 480 is for height
fpsReader = cvzone.FPS()  # gets the FPS of the current frames displayed

#-----------------------MAIN WHILE LOOP----------------------

while True:

    #GETTING FRAMES AND ROI
    success, img = cap.read()
    roi = img[100:400, 300:600]  # height,width
    final_img = cv2.rectangle(img, (300, 100), (600, 400), (255, 0, 0), 2)
    roi = cv2.resize(roi, (150, 150))
    # adds the fps number on the final output image
    _, final_img = fpsReader.update(final_img)
    cv2.imshow("image", final_img)
    img_array = np.expand_dims(roi, axis=0)

    #PREDICTING BASED ON USER GESTURE
    prediction = model.predict([img_array])
    count = 0
    for i in range(len(prediction[0])):
        if (prediction[0][i] == 0):
            count += 1
        elif (prediction[0][i] == 1):
            break
    if(count == 0):
        prediction = 'paper'
    elif(count == 1):
        prediction = 'rock'
    elif(count == 2):
        prediction = 'scissors'

    #ESCAPE CRITERIA - CLICK ESCAPE WHEN TAKING PICTURE
    k = cv2.waitKey(33)
    if k == 27:  # Esc key to stop
        cv2.destroyAllWindows()
        break
    elif k == -1:  # normally -1 returned,so don't print it
        continue

#-----------------------DECIDING FINAL RESULT OF GAME----------------------

#COMPUTER DECIDES ITS CHOICE
comp_choice = random.randint(0, 2)
if(comp_choice == 0):
    comp_final = 'paper'
elif(comp_choice == 1):
    comp_final = 'rock'
else:
    comp_final = 'scissors'
#print("Computer chose : ",comp_choice,comp_final)

#FINAL RESULT DECIDED
if(comp_choice == count):
    res_text = "Draw!"
    print("Draw!")
else:
    if(comp_choice == 1 and count == 0):
        res_text = "You win! Congrats :)"
        print("You win! Congrats :) ")
    elif(comp_choice == 0 and count == 1):
        res_text = "Computer won! Sorry you lost :("
        print("Computer won! Sorry you lost :(")
    elif (comp_choice == 2 and count == 1):
        res_text = "You win! Congrats :)"
        print("You win! Congrats :)")
    elif (comp_choice == 1 and count == 2):
        res_text = "Computer won! Sorry you lost :("
        print("Computer won! Sorry you lost :(")
    elif (comp_choice == 2 and count == 0):
        res_text = "Computer won! Sorry you lost :("
        print("Computer won! Sorry you lost :(")
    elif (comp_choice == 0 and count == 2):
        res_text = "You win! Congrats :)"
        print("You win! Congrats :)")

#FINAL OUTPUT FRAME
roi = cv2.resize(roi, (640, 480))
if(res_text == "You win! Congrats :)"):
    color = (0, 255, 0)
elif(res_text == "Computer won! Sorry you lost :("):
    color = (0, 0, 255)
else:
    color = (255, 0, 0)
comp_disp_text = "Computer chose : "+comp_final
user_disp_text = "You chose : "+prediction
roi = cv2.putText(roi, res_text, (100, 50),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
roi = cv2.putText(roi, user_disp_text, (100, 200),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 186, 122), 2, cv2.LINE_AA)
roi = cv2.putText(roi, comp_disp_text, (100, 350),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (22, 56, 223), 2, cv2.LINE_AA)
cv2.imshow("final roi", roi)
cv2.waitKey(0)
# closing all open windows
cv2.destroyAllWindows()

#---------------------------------------------------------
