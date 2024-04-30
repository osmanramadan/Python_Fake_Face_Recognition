
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import joblib

def calc_hist(img):
    histogram = []
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram.extend(histr.ravel())
    return histogram

modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
clf = joblib.load('models/face_spoofing.pkl')
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    faces = net.forward()

    face_detected = False  # Flag to check if a face is detected

    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            face_detected = True  # Set flag to True if a face is detected
            box = faces[0, 0, i, 3:7] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            (x, y, x1, y1) = box.astype("int")
            roi = img[y:y1, x:x1]

            try:
                img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
                img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)
            except Exception as e:
                print("Error:", e)

            ycrcb_hist = calc_hist(img_ycrcb)
            luv_hist = calc_hist(img_luv)

            # Ensure that histograms have sufficient data
            if len(ycrcb_hist) >= 4 and len(luv_hist) >= 4:
                # Extract only a subset of features from the histograms
                ycrcb_features = ycrcb_hist[:2]
                luv_features = luv_hist[:2]

                # Concatenate the features
                feature_vector = np.concatenate((ycrcb_features, luv_features))

                feature_vector = feature_vector.reshape(1, -1)

                try:
                    prediction = clf.predict_proba(feature_vector)
                    prob = prediction[0][1]

                    if prob >= 0.7:
                        text = "False"
                        color = (0, 0, 255)  # Red color for spoofed face
                    else:
                        text = "True"
                        color = (0, 255, 0)  # Green color for genuine face

                    cv2.rectangle(img, (x, y), (x1, y1), color, 2)
                    cv2.putText(img=img, text=text, org=(x, y - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9,
                                color=color, thickness=2, lineType=cv2.LINE_AA)
                except Exception as e:
                    print("Error:", e)

    # If no face is detected, classify the image as "False"
    if not face_detected:
        cv2.putText(img=img, text="False", org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9,
                    color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow('Face Recognition', img)
    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

