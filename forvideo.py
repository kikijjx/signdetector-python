import tkinter as tk
import cv2
import numpy as np
from tkinter import filedialog
from PIL import Image, ImageTk
import pytesseract
import concurrent.futures
from tkinter import Scale


# Нижняя граница распознавания красного цвета (контура знака)
lower_bound = np.array((142, 75, 50), np.uint8)
# Верхняя граница распознавания красного цвета (контура знака)
upper_bound = np.array((179, 255, 255), np.uint8)

lower_bound_lowlight = np.array((0, 75, 40), np.uint8)

upper_bound_lowlight = np.array((179, 255, 100), np.uint8)

# Подключение tesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Загрузка, ресайз, перевод в HSV и бинаризация эталонного изображения
sample = cv2.imread("etalon.jpg")
sample = cv2.resize(sample, (64,64))
sample = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
sample = cv2.inRange(sample, lower_bound, upper_bound)

def process_contour(contour, cp, font, x, y, w, h, frame):

    area = cv2.contourArea(contour)
    if area < 100:
        return

    roi = cp[y:y+h, x:x+w]
    digits = cv2.cvtColor(roi, cv2.COLOR_HSV2BGR).copy()
    roi = cv2.resize(roi, (64,64))
    roi = cv2.inRange(roi, lower_bound, upper_bound)

    speed_lim = 0
    for i in range(64):
        for j in range(64):
            if roi[i][j] == sample[i][j]:
                speed_lim += 1

    if speed_lim > 3000:
        #roi = digits.copy()
        _, thresh = cv2.threshold(digits, 127, 255, cv2.THRESH_BINARY_INV)
        digits = pytesseract.image_to_string(thresh, config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
        if digits != '' and int(digits) % 10 == 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Speed limit ' + str(int(digits)), (x, y - 5), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            #return
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #cv2.putText(frame, ' ' + str(int(digits)), (x, y - 5), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

def process_frame(frame, current_frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    bottom_half = gray[height // 2:, :]  # Выбрать только нижнюю половину изображения
    median_intensity = np.mean(bottom_half)  # Вычислить медианное значение пикселей
    norm_intensity = median_intensity / 255.0  # Нормализовать уровень освещенности
    #print(f'Intensity: {norm_intensity}')
    # Перевод в HSV, бинаризация и утолщение контуров исходного изображения
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cp = hsv.copy()
    if (norm_intensity < 0.5):
        binary = cv2.inRange(hsv, lower_bound_lowlight, upper_bound_lowlight)
    else:
        binary = cv2.inRange(hsv, lower_bound, upper_bound)

    binary = cv2.dilate(binary, None, iterations=1)

    # Выделение контуров
    #contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    # Шрифт
    font = cv2.FONT_HERSHEY_DUPLEX
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            executor.submit(process_contour, contour, cp, font, x, y, w, h, frame)

    # Отображение обработанного кадра на canvas
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    canvas.image = imgtk
    canvas.create_image(0, 0, image=imgtk, anchor=tk.NW)

def load_video():
    global cap, total_frames, slider

    file_path = filedialog.askopenfilename()

    cap = cv2.VideoCapture(file_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    canvas.config(width=width, height=height)
    slider.config(to=total_frames-1)

    play_video()

def play_video():
    global cap, slider

    while True:
        current_frame = slider.get()
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if not ret:
            break

        process_frame(frame, current_frame)

        slider.set(current_frame + 1)
        root.update()

def on_slider_change(event):
    global cap
    frame_number = slider.get()
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        process_frame(frame, frame_number)

root = tk.Tk()
root.title('Video Player')

slider = Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, length=1440, command=on_slider_change)
slider.pack(side=tk.TOP, padx=10, pady=10)

canvas = tk.Canvas(root, width=640, height=480)
canvas.pack(side=tk.LEFT, padx=10, pady=10)

button = tk.Button(root, text='Load Video', command=load_video)
button.pack(side=tk.RIGHT, padx=10, pady=10)

root.mainloop()
