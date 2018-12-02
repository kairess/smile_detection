import cv2, dlib
from keras.models import load_model
import numpy as np

video_path = 'test/video.mp4'
model = load_model('models/happy_gray_aug2.h5')

detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('%s_output.mp4' % (video_path.split('.')[0]), fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while(cap.isOpened()):
  ret, img = cap.read()

  if not ret:
    break

  # img = cv2.resize(img, dsize=(640, int(img.shape[0] * 640 / img.shape[1])))

  input_img = img.copy()

  faces = detector(img)

  for face in faces:
    padding_size = int(face.width() / 1.2)
    x1, y1, x2, y2 = face.left() - padding_size, face.top() - padding_size, face.right() + padding_size, face.bottom() + padding_size

    desired_size = max(x2 - x1, y2 - y1)

    cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)

    # padding
    ones = np.ones_like(input_img)
    ones[face.top():face.bottom(), face.left():face.right()] = 0
    input_img[np.logical_and(input_img == [[[0,0,0]]], ones.astype(np.bool))] = 140

    x1, y1 = max(0, x1), max(0, y1)
    input_img = input_img[y1:y2, x1:x2]

    delta_w = desired_size - input_img.shape[1]
    delta_h = desired_size - input_img.shape[0]
    top, bottom = delta_h//2, delta_h//2
    left, right = delta_w//2, delta_w//2

    input_img = cv2.copyMakeBorder(input_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(140, 140, 140))

    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    input_img = cv2.resize(input_img, dsize=(64, 64)).astype(np.float64)

    input_img_copy = input_img.copy()
    input_img -= np.mean(input_img_copy, keepdims=True)
    input_img /= (np.std(input_img_copy, keepdims=True) + 1e-6)

    cv2.imshow('input', input_img)

    pred = model.predict(input_img.reshape(1, 64, 64, 1))

    is_smile = pred[0][0] > 0.5

    cv2.putText(img, 'Smile:', (img.shape[1]//2-100, img.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    if is_smile:
      cv2.putText(img, '%s(%s%%)' % (is_smile, int(pred[0][0] * 100)), (img.shape[1]//2, img.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
      cv2.putText(img, '%s(%s%%)' % (is_smile, int(pred[0][0] * 100)), (img.shape[1]//2, img.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    break

  cv2.imshow('img', img)
  out.write(img)
  if cv2.waitKey(1) == ord('q'):
    break