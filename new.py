import cv2
from tkinter import *
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import numpy as np
import pyperclip

__author__ = "Teeraphat Kullanankanjana"
__version__ = "0.1.0"

class HSVRangeFinder:
    def __init__(self):
        self.image_path = None
        self.image = None
        self.processed_image = None
        self.mask = None

        # Create the main tkinter window
        self.window = Tk()
        self.window.geometry('3000x900')
        self.window.title('HSV Range Finder')
        self.window.resizable(0, 0)

        # --- Camera Frames ---
        self.mainCameraFrame = LabelFrame(self.window, text='Main Image')
        self.mainCameraFrame.place(x=0, y=0)

        self.vidLabel1 = Label(self.mainCameraFrame)
        self.vidLabel1.configure(width=1000, height=800)
        self.vidLabel1.pack()

        self.contourCameraFrame = LabelFrame(self.window, text='Result Image')
        self.contourCameraFrame.place(x=1005, y=0)

        self.vidLabel2 = Label(self.contourCameraFrame)
        self.vidLabel2.configure(width=1000, height=800)
        self.vidLabel2.pack()

        self.outCameraFrame = LabelFrame(self.window, text='Binary Mask')
        self.outCameraFrame.place(x=2000, y=0)

        self.vidLabel3 = Label(self.outCameraFrame)
        self.vidLabel3.configure(width=1000, height=800)
        self.vidLabel3.pack()

        # --- Image Control Frame ---
        self.imageControlFrame = LabelFrame(self.window, text='Image Control')
        self.imageControlFrame.place(x=0, y=425)

        self.uploadBtn = Button(self.imageControlFrame, text='открыть', command=self.upload_image)
        self.uploadBtn.grid(row=0, column=0)

        # --- Slider Section ---
        self.l_h, self.l_s, self.l_v = DoubleVar(), DoubleVar(), DoubleVar()
        self.u_h, self.u_s, self.u_v = DoubleVar(), DoubleVar(), DoubleVar()

        self.u_h.set(179)
        self.u_s.set(255)
        self.u_v.set(255)

        def get_lh():
            return '{:.0f}'.format(self.l_h.get())

        def lh_changed(event):
            self.lhShow.configure(text=get_lh())
            self.update_image()

        def get_ls():
            return '{:.0f}'.format(self.l_s.get())

        def ls_changed(event):
            self.lsShow.configure(text=get_ls())
            self.update_image()

        def get_lv():
            return '{:.0f}'.format(self.l_v.get())

        def lv_changed(event):
            self.lvShow.configure(text=get_lv())
            self.update_image()

        def get_uh():
            return '{:.0f}'.format(self.u_h.get())

        def uh_changed(event):
            self.uhShow.configure(text=get_uh())
            self.update_image()

        def get_us():
            return '{:.0f}'.format(self.u_s.get())

        def us_changed(event):
            self.usShow.configure(text=get_us())
            self.update_image()

        def get_uv():
            return '{:.0f}'.format(self.u_v.get())

        def uv_changed(event):
            self.uvShow.configure(text=get_uv())
            self.update_image()

        self.sliderFrame = LabelFrame(self.window, text='HSV Range Adjustment')
        self.sliderFrame.place(x=1185, y=725)

        self.lhLabel = Label(self.sliderFrame, text='Lower Hue:')
        self.lhLabel.grid(row=0, column=0)
        self.lhSlider = Scale(self.sliderFrame, orient='horizontal', from_=0, to=179, command=lh_changed,
                              variable=self.l_h)
        self.lhSlider.grid(row=0, column=1)

        self.lsLabel = Label(self.sliderFrame, text='Lower Saturation:')
        self.lsLabel.grid(row=0, column=3)
        self.lsSlider = Scale(self.sliderFrame, orient='horizontal', from_=0, to=255, command=ls_changed,
                              variable=self.l_s)
        self.lsSlider.grid(row=0, column=4)

        self.lvLabel = Label(self.sliderFrame, text='Lower Value:')
        self.lvLabel.grid(row=0, column=5)
        self.lvSlider = Scale(self.sliderFrame, orient='horizontal', from_=0, to=255, command=lv_changed,
                              variable=self.l_v)
        self.lvSlider.grid(row=0, column=6)

        self.uhLabel = Label(self.sliderFrame, text='Upper Hue:')
        self.uhLabel.grid(row=1, column=0)
        self.uhSlider = Scale(self.sliderFrame, orient='horizontal', from_=0, to=179, command=uh_changed,
                              variable=self.u_h)
        self.uhSlider.grid(row=1, column=1)

        self.usLabel = Label(self.sliderFrame, text='Upper Saturation:')
        self.usLabel.grid(row=1, column=3)
        self.usSlider = Scale(self.sliderFrame, orient='horizontal', from_=0, to=255, command=us_changed,
                              variable=self.u_s)
        self.usSlider.grid(row=1, column=4)

        self.uvLabel = Label(self.sliderFrame, text='Upper Value:')
        self.uvLabel.grid(row=1, column=5)
        self.uvSlider = Scale(self.sliderFrame, orient='horizontal', from_=0, to=255, command=uv_changed,
                              variable=self.u_v)
        self.uvSlider.grid(row=1, column=6)

        self.resultFrame = LabelFrame(self.window, text='Get Result')
        self.resultFrame.place(x=745, y=725)

        self.lrLabel = Label(self.resultFrame, text='HSV Lower Range')
        self.lrLabel.grid(row=0, column=0, columnspan=3)

        self.lhShow = Label(self.resultFrame, text='0')
        self.lhShow.grid(row=1, column=0)
        self.lsShow = Label(self.resultFrame, text='0')
        self.lsShow.grid(row=1, column=1)
        self.lvShow = Label(self.resultFrame, text='0')
        self.lvShow.grid(row=1, column=2)

        self.urLabel = Label(self.resultFrame, text='HSV Upper Range')
        self.urLabel.grid(row=2, column=0, columnspan=3)

        self.uhShow = Label(self.resultFrame, text='0')
        self.uhShow.grid(row=3, column=0)
        self.usShow = Label(self.resultFrame, text='0')
        self.usShow.grid(row=3, column=1)
        self.uvShow = Label(self.resultFrame, text='0')
        self.uvShow.grid(row=3, column=2)

        self.cpyupperBtn = Button(self.resultFrame, text='Copy', command=self.get_lowerRange)
        self.cpyupperBtn.grid(row=0, column=3, rowspan=3)

        self.cpylowwerBtn = Button(self.resultFrame, text='Copy', command=self.get_upperRange)
        self.cpylowwerBtn.grid(row=3, column=3, rowspan=3)

    def upload_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.image = cv2.imread(self.image_path)
            self.update_image()

    def get_lowerRange(self):
        lowerRange = '{},{},{}'.format(self.get_lh(), self.get_ls(), self.get_lv())
        pyperclip.copy(lowerRange)

    def get_upperRange(self):
        upperRange = '{},{},{}'.format(self.get_uh(), self.get_us(), self.get_uv())
        pyperclip.copy(upperRange)

    def update_image(self):
        if self.image is None:
            return

        lower_bound = np.array([self.l_h.get(), self.l_s.get(), self.l_v.get()])
        upper_bound = np.array([self.u_h.get(), self.u_s.get(), self.u_v.get()])

        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.mask = cv2.inRange(hsv, lower_bound, upper_bound)
        self.mask = cv2.dilate(self.mask, None, iterations=1)

        self.processed_image = cv2.bitwise_and(self.image, self.image, mask=self.mask)

        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        img1 = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        img1 = Image.fromarray(img1)
        img1 = ImageTk.PhotoImage(image=img1)

        img2 = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(img2)
        img2 = ImageTk.PhotoImage(image=img2)

        img3 = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        img3 = Image.fromarray(img3)
        img3 = ImageTk.PhotoImage(image=img3)

        self.vidLabel1.config(image=img1)
        self.vidLabel1.image = img1

        self.vidLabel2.config(image=img2)
        self.vidLabel2.image = img2

        self.vidLabel3.config(image=img3)
        self.vidLabel3.image = img3

    def get_lh(self):
        return '{:.0f}'.format(self.l_h.get())

    def lh_changed(self, event):
        self.lhShow.configure(text=self.get_lh())
        self.update_image()

    def get_ls(self):
        return '{:.0f}'.format(self.l_s.get())

    def ls_changed(self, event):
        self.lsShow.configure(text=self.get_ls())
        self.update_image()

    def get_lv(self):
        return '{:.0f}'.format(self.l_v.get())

    def lv_changed(self, event):
        self.lvShow.configure(text=self.get_lv())
        self.update_image()

    def get_uh(self):
        return '{:.0f}'.format(self.u_h.get())

    def uh_changed(self, event):
        self.uhShow.configure(text=self.get_uh())
        self.update_image()

    def get_us(self):
        return '{:.0f}'.format(self.u_s.get())

    def us_changed(self, event):
        self.usShow.configure(text=self.get_us())
        self.update_image()

    def get_uv(self):
        return '{:.0f}'.format(self.u_v.get())

    def uv_changed(self, event):
        self.uvShow.configure(text=self.get_uv())
        self.update_image()

    def cleanup(self):
        self.window.destroy()

    def run(self):
        self.window.protocol("WM_DELETE_WINDOW", self.cleanup)
        self.window.mainloop()


if __name__ == '__main__':
    app = HSVRangeFinder()
    app.run()
