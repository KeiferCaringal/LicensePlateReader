import cv2
import os
import math
import easyocr
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import logging

# Set TensorFlow log level to suppress unnecessary messages
tf.get_logger().setLevel(logging.ERROR)


class LicensePlateDetectionAndOCR:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Detection and OCR")

        self.cap = None
        self.camera_running = False

        # Load YOLO model for object detection
        self.model = YOLO("best.pt")
        self.classNames = ["plate"]  # Class names used in YOLO model

        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['en'], gpu=True)

        # Path to save captured images
        self.save_path = "saved_images"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Create main frame for GUI
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack()

        # Button to start/stop camera feed
        self.start_button = tk.Button(self.main_frame, text="Start Camera", command=self.toggle_camera)
        self.start_button.pack(pady=10)

        # Button to exit program
        self.exit_button = tk.Button(self.main_frame, text="Exit", command=self.exit_program)
        self.exit_button.pack(side=tk.BOTTOM, pady=10, padx=10)

        # Label to display video feed
        self.video_label = tk.Label(self.main_frame)
        self.video_label.pack()

        # Name of the window
        self.window_name = 'License Plate Detection and OCR'

    # Method to toggle camera feed
    def toggle_camera(self):
        if not self.camera_running:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(3, 640)  # Set width of the video capture
            self.cap.set(4, 480)  # Set height of the video capture
            self.camera_running = True
            self.start_button.config(text="Stop Camera")
            self.show_camera_feed()
        else:
            self.camera_running = False
            self.cap.release()
            self.start_button.config(text="Start Camera")

    # Method to continuously display camera feed
    def show_camera_feed(self):
        if self.camera_running:
            ret, frame = self.cap.read()
            if ret:
                frame = self.process_frame(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = img
                self.video_label.configure(image=img)
                self.video_label.after(10, self.show_camera_feed)

    # Method to process each frame from the camera feed
    def process_frame(self, img):
        results = self.model(img, save_crop=True, conf=0.5)  # Perform object detection using YOLO

        plate_texts = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                width = x2 - x1
                height = y2 - y1
                x1 = max(0, x1 - width // 4)
                x2 = min(img.shape[1], x2 + width // 4)
                y1 = max(0, y1 - height // 4)
                y2 = min(img.shape[0], y2 + height // 4)
                plate_img = img[y1:y2, x1:x2]
                text = self.perform_ocr_on_image(plate_img)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(img, f"{self.classNames[cls]}: {text}", org, font, fontScale, color, thickness)
                plate_texts.append(text)

        for i, text in enumerate(plate_texts):
            cv2.putText(img, f"Plate {i+1}: {text}", (20, 30 * (i+1)), font, 0.7, (0, 255, 0), 2)

        return img

    # Method to perform OCR on an image
    def perform_ocr_on_image(self, img):
        try:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            results = self.reader.readtext(gray_img)
            text = ""
            if len(results) == 1:
                text = results[0][1]
            elif results:
                for res in results:
                    if res[2] > 0.2:
                        text = res[1]
                        break
            return text
        except Exception as e:
            print(f"Error performing OCR: {e}")

    # Method to exit the program
    def exit_program(self):
        if messagebox.askokcancel("Exit", "Do you want to exit?"):
            if self.cap:
                self.cap.release()
            self.root.destroy()


# Main program entry point
if __name__ == "__main__":
    root = tk.Tk()  # Create Tkinter window
    app = LicensePlateDetectionAndOCR(root)  # Create instance of LicensePlateDetectionAndOCR class
    root.mainloop()  # Start the GUI event loop
