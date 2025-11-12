# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows

## Name:  S.Harika 
## Reg No: 212224240155
## Program:
```
import cv2
import matplotlib.pyplot as plt
import numpy as np
```
```
w_glass = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)
wo_glass = cv2.imread('image2.png', cv2.IMREAD_GRAYSCALE)
group = cv2.imread('image3.png', cv2.IMREAD_GRAYSCALE)
```
```
w_glass1 = cv2.resize(w_glass, (1000, 1000))
wo_glass1 = cv2.resize(wo_glass, (1000, 1000)) 
group1 = cv2.resize(group, (1000, 1000))
```
```
plt.figure(figsize=(15,10))
plt.subplot(1,3,1);plt.imshow(w_glass1,cmap='gray');plt.title('With Glasses');plt.axis('off')
plt.subplot(1,3,2);plt.imshow(wo_glass1,cmap='gray');plt.title('Without Glasses');plt.axis('off')
plt.subplot(1,3,3);plt.imshow(group1,cmap='gray');plt.title('Group Image');plt.axis('off')
plt.show()
```
```
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def detect_and_display(image):
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 10)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
```
```
import cv2
from matplotlib import pyplot as plt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error: Cascade file not loaded properly!")
else:
    print("Cascade loaded successfully.")
w_glass1 = cv2.imread('image1.png')  # <-- replace with your image filename

if w_glass1 is None:
    print("Error: Image not found. Check the filename or path.")
else:
    print("Image loaded successfully.")
def detect_and_display(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
    
    return image
if w_glass1 is not None and not face_cascade.empty():
    result = detect_and_display(w_glass1)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
```
```
import cv2
from matplotlib import pyplot as plt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
if face_cascade.empty():
    print("Error: Face cascade not loaded properly!")
if eye_cascade.empty():
    print("Error: Eye cascade not loaded properly!")
# (Change the filenames as per your actual image files)
w_glass = cv2.imread('image1.png')
wo_glass = cv2.imread('image2.png')
group = cv2.imread('image3.png')
def detect_eyes(image):
    face_img = image.copy()
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(face_img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    
    return face_img
if w_glass is not None:
    w_glass_result = detect_eyes(w_glass)
    plt.imshow(cv2.cvtColor(w_glass_result, cv2.COLOR_BGR2RGB))
    plt.title("With Glasses - Eye Detection")
    plt.axis("off")
    plt.show()

if wo_glass is not None:
    wo_glass_result = detect_eyes(wo_glass)
    plt.imshow(cv2.cvtColor(wo_glass_result, cv2.COLOR_BGR2RGB))
    plt.title("Without Glasses - Eye Detection")
    plt.axis("off")
    plt.show()

if group is not None:
    group_result = detect_eyes(group)
    plt.imshow(cv2.cvtColor(group_result, cv2.COLOR_BGR2RGB))
    plt.title("Group - Eye Detection")
    plt.axis("off")
    plt.show()
```
```
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("on")
    plt.title("Video Face Detection")
    plt.show()
    break

cap.release()
```
```
from IPython.display import clear_output, display
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
```
```
def new_detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return frame
```
```
video_capture = cv2.VideoCapture(0)
captured_frame = None   

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("No frame captured from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = new_detect(gray, frame)
    clear_output(wait=True)
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Video - Face & Eye Detection")
    display(plt.gcf())
    captured_frame = canvas.copy()  
    break
```
```
video_capture.release()
if captured_frame is not None and captured_frame.size > 0:
    cv2.imwrite('captured_face_eye.png', captured_frame)
    captured_image = cv2.imread('captured_face_eye.png', cv2.IMREAD_GRAYSCALE)
    plt.imshow(captured_image, cmap='gray')
    plt.title('Captured Face with Eye Detection')
    plt.axis('off')
    plt.show()
else:
    print("No valid frame to save.")
```
```
image = cv2.imread('image4.png') 
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0) 

edges = cv2.Canny(blurred_image, 50, 150)  

plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')
```
```
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

result_image = image.copy() 
for contour in contours:
    if cv2.contourArea(contour) > 50: 
        x, y, w, h = cv2.boundingRect(contour)  
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  


plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title("Handwriting Detection")
plt.axis('off')
```

## OUTPUT:


<img width="417" height="446" alt="Screenshot 2025-11-12 161016" src="https://github.com/user-attachments/assets/832e4905-7160-45b7-bf47-0fd3709b7a1e" />


<img width="444" height="445" alt="Screenshot 2025-11-12 161038" src="https://github.com/user-attachments/assets/f5dd5ea3-852c-435b-bf9d-23c18c1278c4" />


<img width="432" height="436" alt="Screenshot 2025-11-12 161100" src="https://github.com/user-attachments/assets/60bca4ad-44fc-4e95-94a7-f0908857290e" />


<img width="497" height="526" alt="Screenshot 2025-11-12 161123" src="https://github.com/user-attachments/assets/3c93e822-2c72-4223-90fe-0d81f540a6e9" />


<img width="506" height="524" alt="Screenshot 2025-11-12 161204" src="https://github.com/user-attachments/assets/bcd1f527-4fcf-4309-92ea-d1ec23564f9e" />


<img width="378" height="514" alt="Screenshot 2025-11-12 161229" src="https://github.com/user-attachments/assets/26f00e94-fc5b-4eab-a512-6c668bfc7bc0" />


<img width="394" height="504" alt="Screenshot 2025-11-12 161334" src="https://github.com/user-attachments/assets/0edb7328-25b6-4575-99b3-857b4a90c219" />


<img width="722" height="542" alt="Screenshot 2025-11-12 161357" src="https://github.com/user-attachments/assets/13a054c4-4114-443d-ab52-5bef67c73d80" />


<img width="725" height="514" alt="Screenshot 2025-11-12 161419" src="https://github.com/user-attachments/assets/0dfc72d3-8fb8-49e8-af34-4f13a81d58a3" />


<img width="648" height="514" alt="Screenshot 2025-11-12 161457" src="https://github.com/user-attachments/assets/d09bd1a1-28dc-48b0-b627-019915a5c0fe" />


<img width="667" height="515" alt="Screenshot 2025-11-12 161517" src="https://github.com/user-attachments/assets/30bab4e2-ae95-4438-a33d-b8432c157d97" />


<img width="699" height="368" alt="Screenshot 2025-11-12 161536" src="https://github.com/user-attachments/assets/28887876-e541-4e05-bfbc-0b0b38ce2bb5" />


<img width="766" height="387" alt="Screenshot 2025-11-12 161554" src="https://github.com/user-attachments/assets/6291a801-3680-42fb-b7b1-cf4975fa0ccb" />



