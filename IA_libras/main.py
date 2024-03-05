#import tensorflow as tf
from tensorflow import keras
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np

camera = cv2.VideoCapture(0) #Abrimos a camera

hands = mp.solutions.hands.Hands(max_num_hands=1) #função mãos do midiapipe

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
model = load_model('modeloSinais.h5') #carregando as fotos
data = np.ndarray(shape=(1,224,224,3), dtype=np.float32) # definimos o array de entrada

while True: #Enquanto for verdade
    sucesso, imagem = camera.read()
    frameRGB = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    resultados = hands.process(frameRGB)
    handsPoints = resultados.multi_hand_landmarks
    h, w, _ = imagem.shape

    if handsPoints != None:
     for hand in handsPoints:
        x_max = 0
        y_max = 0
        x_min = w
        y_min = h
        for lm in hand.landmark: # criando um retangulo
            x,y =int(lm.x*w), int(lm.y*h)
            if x>x_max:
                x_max = x
            if x<x_min:
                x_min = x
            if y > y_max:
                y_max = y
            if y < y_max:
                y_min = y
        cv2.rectangle(imagem, (x_min-50, y_min-50), (x_max+50, y_max+50),(0,255,0) , 2)

        try:
            imgCorte = imagem[y_min-50:y_max +50, x_min-50:x_max+50] #realiza o corte da imagem para ser passado apenas a mão para a predição
            imgCorte = cv2.resize(imgCorte, (224,224)) # resimensionando a imagem
            imgArray = np.asarray(imgCorte)        # tranformando em array
            normalized_image_array = (imgArray.astype(np.float32)/127) -1 # normaliza a imagem
            data[0] = normalized_image_array # adiciona em um vetor a imagem normalizada
            predict = model.predict(data) # retorna a probabilidade de cada uma das nossas classes
            indexVal = np.argmax(predict) #retira a classe com maior porcentagem

            cv2.putText(imagem, classes[indexVal], (x_min-50,y_min-65), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,255),5) # inserir texto dentro da imagem
        except:
            continue

    cv2.imshow('Imagem', imagem)
    cv2.waitKey(1)