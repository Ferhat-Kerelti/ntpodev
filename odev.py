import cv2
import face_recognition

# Video kaynağını başlat (webcam)
video_capture = cv2.VideoCapture(0)

# Yüzleri tanımak için tanınan yüz verilerini yükleyin
known_face_encodings = []
known_face_names = []

# Kendi resminizden veya başka bir kaynaktan yüz verisi eklemek
image = face_recognition.load_image_file("your_image.jpg")
face_encoding = face_recognition.face_encodings(image)[0]
known_face_encodings.append(face_encoding)
known_face_names.append("Your Name")

while True:
    # Video akışından bir frame alın
    ret, frame = video_capture.read()

    # Görüntüyü RGB formatına çevirin (OpenCV BGR kullanır)
    rgb_frame = frame[:, :, ::-1]

    # Frame'deki yüzleri tespit et
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Yüz tanımlamaları yap
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Eğer eşleşme varsa, ilk eşleşen yüzü kullan
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Yüz üzerine kutu çizme
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Yüzün adı yazdırma
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Görüntüyü ekranda göster
    cv2.imshow('Video', frame)

    # 'q' tuşuna basarak çıkış yapın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Video kaynağını serbest bırakın
video_capture.release()
cv2.destroyAllWindows()
