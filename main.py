import cv2
import face_recognition

image1 = face_recognition.load_image_file('image1.jpeg')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
locationface1 = face_recognition.face_locations(image1)[0]
faceencode1 = face_recognition.face_encodings(image1)[0]
facebox1 = cv2.rectangle(image1, (locationface1[3], locationface1[0]), (locationface1[1], locationface1[2]), (255,0,0),2)

image2 = face_recognition.load_image_file('image3.jpeg')
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
locationface2 = face_recognition.face_locations(image2)[0]
faceencode2 = face_recognition.face_encodings(image2)[0]
facebox2 = cv2.rectangle(image2, (locationface2[3], locationface2[0]), (locationface2[1], locationface2[2]), (255,0,0),2)

result = face_recognition.compare_faces([faceencode1], faceencode2)
similarity = face_recognition.face_distance([faceencode1], faceencode2)

print(result, similarity)

cv2.imshow('image1', facebox1)
cv2.imshow('image2', facebox2)
cv2.waitKey(0)
