import tkinter as tk
import cv2
import numpy as np
from tkinter import filedialog
from PIL import Image, ImageTk
import pytesseract
import concurrent.futures
from tkinter import Scale
print(cv2.cuda.getCudaEnabledDeviceCount())
# Нижняя граница распознавания красного цвета (контура знака)
lower_bound = np.array((142, 75, 50), np.uint8)
# Верхняя граница распознавания красного цвета (контура знака)
upper_bound = np.array((179, 255, 255), np.uint8)
# Подключение tesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Загрузка, ресайз, перевод в HSV и бинаризация эталонного изображения
sample = cv2.imread("etalon.jpg")
sample = cv2.resize(sample, (64, 64))
sample = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
sample = cv2.inRange(sample, lower_bound, upper_bound)

def process_contour(contour, cp, font, x, y, w, h, frame):
    area = cv2.contourArea(contour)
    if area < 100:
        return

    roi = cp[y:y + h, x:x + w]
    digits = cv2.cvtColor(roi, cv2.COLOR_HSV2BGR).copy()
    roi = cv2.resize(roi, (64, 64))
    roi = cv2.inRange(roi, lower_bound, upper_bound)

    speed_lim = 0
    for i in range(64):
        for j in range(64):
            if roi[i][j] == sample[i][j]:
                speed_lim += 1

    if speed_lim > 3000:
        gray = cv2.cvtColor(digits, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        digits = pytesseract.image_to_string(thresh, config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
        if digits != '' and int(digits) % 10 == 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Speed limit ' + str(int(digits)), (x, y - 5), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

def process_frame(frame, current_frame):
    # Перевод в HSV с использованием CUDA
    hsv = cv2.cuda.cvtColor(cv2.cuda_GpuMat(frame), cv2.COLOR_BGR2HSV)
    cp = hsv.copy()
    binary = cv2.cuda.inRange(hsv, lower_bound, upper_bound)
    binary = cv2.cuda.dilate(binary, None, iterations=1)

    # Перемещение данных с GPU на CPU для findContours
    binary = binary.download()
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Шрифт
    font = cv2.FONT_HERSHEY_DUPLEX
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for contour in contours:
            x, y, w, h = cv2
