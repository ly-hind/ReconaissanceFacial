import cv2,numpy as np,face_recognition
import streamlit as st
from PIL import Image
 
#import_Signatures
signatures_class=np.load('FaceSignatures_db.npy')
X=signatures_class[ : , 0: -1].astype('float')
Y=signatures_class[ : , -1]

# Barre latérale pour télécharger une image
st.sidebar.header("Téléchargez une image")
img = st.sidebar.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])

if img  is not None :
     img = Image.open(img)
     img_numpy = np.array(img)
     resized_image =cv2.resize(img_numpy, (0,0), None, 0.25, 0.25)

     #resized_image = img_numpy((400, 300))
     resized_image = cv2.cvtColor( resized_image, cv2.COLOR_BGR2RGB)
     #facesCurrent représente les coordonnées des visages détectés.
     facesCurrent = face_recognition.face_locations(resized_image)

    #Extraction des caractéristiques faciales 
     encodesCurrent = face_recognition.face_encodings(resized_image, facesCurrent)
    
    #Comparaison des caractéristiques faciales 
     for encodeFace, faceLoc in zip(encodesCurrent, facesCurrent):
#déterminer si le visage correspond à un visage dans la base de données X
            matches = face_recognition.compare_faces(X, encodeFace)
 #calcule la distance entre les encodages actuels et ceux de la base de données
            faceDis = face_recognition.face_distance(X, encodeFace)
# obtient l'indice du visage le plus proche, c'est-à-dire celui avec la distance la plus faible.
            matchIndex = np.argmin(faceDis)
#cela signifie qu'il y a une correspondance
            if matches[matchIndex]:
             st.write(matches)
             st.write(faceDis)
             st.write(matchIndex)
             name = Y[matchIndex].upper()
             st.write(name)
             pht='./images/'+name+'.jpg'
             st.image(pht)
            