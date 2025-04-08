import cv2.data
import torch
from models.Resnet import ResEmoteNet
from pathlib import Path
from torchvision import transforms
import cv2
import torch.nn.functional as F
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Configure Pytorch computation backend to either Nvidia GPUs or Apple silicon
device = "cuda" if torch.cuda.is_available() else "mps"

# Set up our model
model_path = Path("result/best_model.pth")
model = ResEmoteNet().to(device)
model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
model.eval()

# ============= Build a data transformation pipeline ============
transform = transforms.Compose([
    # Faciliate the execution of SE Block
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load YOLOv8 model for face detection
# Use a model specifically trained for face detection
face_detector = YOLO('result/yolov8n-face.pt') 

# Settings for text
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.2
font_color = (0, 255, 0)
thickness = 3
line_type = cv2.LINE_AA

# Define a list of emotions based on its corresponding index position
emotions = ['angry', 'frustration', 'boredom', 'happy', 'sad', 'surprise', 'neutral']

max_emotion = ''
def infer_emotion(frame):
    """
        Compute the probabilities of different emotions in an image frame
    """
    frame_tensor = transform(frame).unsqueeze(0).to(device)

    # Run inference on image tensor without gradient computation
    with torch.no_grad():
        output = model(frame_tensor)
        probs = F.softmax(output, dim=1)

    # Convert tensors to numpy values
    scores = probs.cpu().numpy().flatten()
    rounded_scores = [round(score, 2) for score in scores]
    return rounded_scores

def infer_max_emotion(x, y, w, h, frame):
    """
        Employ our ML model to infer the most likely emotion in a video frame
        
        x,y: coordinates of top left point
        w,h: width and height of an image frame
        frame: a video frame
    """
    cropped_img = frame[y:y+h, x:x+w]
    pil_img = Image.fromarray(cropped_img)
    scores = infer_emotion(pil_img)
    max_idx = np.argmax(scores)
    return emotions[max_idx]

def print_max_emotion(x, y, frame, emotion):
    """
        Display the most likely emotion label on video frames

        emotion: string label to annote a frame
    """
    org = (x, y - 10)
    cv2.putText(frame, emotion, org, font, font_scale, font_color, thickness, line_type)

def print_all_emotion(x, y, w, h, frame):
    """
        Display all emotion labels and their probability score on video frames
    """
    cropped_img = frame[y:y+h, x:x+w]
    pil_img = Image.fromarray(cropped_img)
    all_scores = infer_emotion(pil_img)
    org = (x+w+10, y-20)

    for k, v in enumerate(emotions):
        text = f"Label {v}: {all_scores[k]}"
        y = org[1] + 40
        org = (org[0], y)
        cv2.putText(frame, text, org, font, font_scale, font_color, thickness, line_type)

def detect_bounding_box(frame, counter):
    """
        Detect faces in a video stream using YOLOv8

        frame: an image frame
        counter: an integer controlling when to invoke our ML model
    """
    global max_emotion
    
    # Run YOLOv8 detection with confidence threshold
    results = face_detector(frame, conf=0.5)  # Higher confidence threshold for more precise detections
    
    # Process each detected face
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates (convert to integers)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Ensure the bounding box is not too large (focus on face)
            # If the box is too large, try to make it more square and centered
            w = x2 - x1
            h = y2 - y1
            
            # If the box is too large, adjust it to be more face-like
            if w > frame.shape[1] // 2 or h > frame.shape[0] // 2:
                # Calculate center of the box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Make the box more square and smaller
                size = min(w, h) // 2
                x1 = max(0, center_x - size)
                y1 = max(0, center_y - size)
                x2 = min(frame.shape[1], center_x + size)
                y2 = min(frame.shape[0], center_y + size)
                
                # Recalculate width and height
                w = x2 - x1
                h = y2 - y1
            
            # Draw a bounding box around a face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Infer emotion on defined interval
            if counter == 0:
                max_emotion = infer_max_emotion(x1, y1, w, h, frame)

            print_max_emotion(x1, y1, frame, max_emotion)
            print_all_emotion(x1, y1, w, h, frame)
    
    return len(results) > 0  # Return True if faces were detected

# Access live webcam feed
video_capture = cv2.VideoCapture(0)

counter = 0
detect_frequency = 5
# Face detection loop
while True:
    ret, video_frame = video_capture.read()

    if ret is False:
        break

    # Draw a bounding box around faces
    faces_detected = detect_bounding_box(video_frame, counter)

    cv2.imshow("Emotion Detection", video_frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

    counter += 1
    if counter == detect_frequency:
        counter = 0

# Video feed cleanup
video_capture.release()
cv2.destroyAllWindows()