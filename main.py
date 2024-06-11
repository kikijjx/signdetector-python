import cv2
import numpy as np
import pytesseract


#нижняя граница распознавания красного цвета (контура знака)
lower_bound = np.array((0, 75, 100), np.uint8)
#lower_bound = np.array((0, 165, 75), np.uint8)
#верхняя граница распознавания красного цвета (контура знака)
upper_bound = np.array((179, 255, 255), np.uint8)
#upper_bound = np.array((179, 255, 135), np.uint8)


#lower_bound = np.array((120, 75, 0), np.uint8)
#upper_bound = np.array((179, 255, 120), np.uint8)


lower_bound_digits = np.array((0, 0, 0), np.uint8)
upper_bound_digits = np.array((179, 86, 75), np.uint8)
#подключение tesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

#загрузка, ресайз, перевод в hsv и бинаризация эталонного изображения
sample = cv2.imread("etalon.jpg")
sample = cv2.resize(sample, (64,64))

sample = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)

sample = cv2.inRange(sample, lower_bound, upper_bound)
cv2.imshow('sample', sample)
cv2.imwrite('samplebinary.jpg', sample)

#загрузка, перевод в hsv, бинаризация и утолщение контуров исходного изображения

image = cv2.imread('4.jpg')

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv', hsv)
cp = hsv.copy()
binary = cv2.inRange(hsv, lower_bound, upper_bound)
binary = cv2.dilate(binary, None, iterations=1)
cv2.imshow('binary', binary)


#выделение контуров
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#cv2.drawContours(image, contours, -1, (255,0,255), 3) #отрисовка контуров на исходном изображении

#шрифт
font = cv2.FONT_HERSHEY_DUPLEX

for contour in contours:

    #если площадь контура меньше 100, то не считаем за знак
    area = cv2.contourArea(contour)
    if area < 100:
        continue

    #вырезаем подозрительный прямоугольник, ресайзим и бинаризируем
    (x, y, w, h) = cv2.boundingRect(contour)
    roi = cp[y:y+h, x:x+w]
    digits = cv2.cvtColor(roi, cv2.COLOR_HSV2BGR).copy()
    roi = cv2.resize(roi, (64,64))


    roi = cv2.inRange(roi, lower_bound, upper_bound)

    #считаем по пикселям схожесть подозрительного прямоугольника и эталона
    speed_lim = 0
    for i in range(64):
        for j in range(64):
            if roi[i][j] == sample[i][j]:
                speed_lim += 1

    #если схожих пикселей больше 3000 то продолжаем проверку (ищем внутри цифры)
    if speed_lim > 3000:
        gray=digits
        #gray = cv2.inRange(digits, lower_bound_digits, upper_bound_digits)

        #gray = cv2.cvtColor(digits, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('gray' + str(speed_lim), gray)

        roi = digits.copy()
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        digits = pytesseract.image_to_string(thresh,
                                             config='--psm 1 --oem 3 -c tessedit_char_whitelist=00123345566778899')
        #если цифры заканчиваются на 0, то этот знак искомый
        if digits != '' and int(digits) % 10 == 0:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, 'Speed limit ' + str(int(digits)), (x, y - 5), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            #cv2.putText(image, str(speed_lim), (x, y - 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('roi', roi)
#
        else:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            #cv2.putText(image, '0', (x, y-5), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            #cv2.putText(image, str(speed_lim), (x, y-20), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.putText(image, 'Speed limit ', (x, y - 5), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        #cv2.imshow('roi', roi)




#cv2.imshow('маска', binary)
cv2.imshow('kartinka', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
