import cv2.data
import torch
from Resnet import ResEmoteNet
from pathlib import Path
from torchvision import transforms
import cv2

# Configure Pytorch computation backend to either Nvidia GPUs or Apple silicon
device = "cuda" if torch.cuda.is_available() else "mps"

# Set up our model
model_path = Path("result/best_model.pth")
model = ResEmoteNet.to(device)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# ============= Build a data transformation pipeline ============
transform = transforms.Compose([
    # Faciliate the execution of SE Block
    transforms.Grayscale(num_output_channels=3),
    # Introduces variability
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Instantiate a haar cascade classifier with a pretrained face detection model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades 
                                        + "haarcascade_frontalface_default.xml")

# Settings for text
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.2
font_color = (0, 255, 0)
thickness = 3
line_type = cv2.LINE_AA

def detect_bounding_box(frame, counter):
    """
        Detect a bounding box around faces in an image frame

        frame: an image frame
        counter: an integer controlling when to invoke our ML model
    """
    pass

# Access live webcam feed
video_capture =cv2.VideoCapture(0)

counter = 0
detect_frequency = 5
# Face detection loop
while True:
    ret, video_frame = video_capture.read()

    if ret is False:
        break

    # Draw a bounding box around faces
    faces = detect_bounding_box(video_frame, counter)

    cv2.imshow("Emotion Dection", video_frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

    counter += 1
    if counter == detect_frequency:
        counter = 0

# Video feed cleanup
video_capture.release()
cv2.destroyAllWindows()