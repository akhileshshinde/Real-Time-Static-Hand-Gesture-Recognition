import torch
import cv2
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression
from utils.plots import Annotator

# Load model
device = select_device(0)
model = DetectMultiBackend(
    "/home/akhilesh/yolov5_new/yolov5/runs/train/hand_sign/weights/best.pt",
    device=device,
    dnn=False
)
model.warmup()

cap = cv2.VideoCapture(0)  
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thres = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    frame_proc = cv2.cvtColor(thres, cv2.COLOR_GRAY2BGR)  
    # Prepare image for model
    img = cv2.resize(frame_proc, (640, 640))
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)

    # Run detection
    pred = model(img)
    pred = non_max_suppression(pred)

    annotator = Annotator(frame)
    for det in pred:
        if len(det):

            scale_factors = torch.tensor(
                [
                    frame.shape[1] / img.shape[3],
                    frame.shape[0] / img.shape[2],
                    frame.shape[1] / img.shape[3],
                    frame.shape[0] / img.shape[2],
                ],
                device=det.device
            )

            det[:, :4] *= scale_factors
            det[:, :4] = det[:, :4].round()

            for *xyxy, conf, cls in det:
                if conf > 0.70:
                    annotator.box_label(xyxy, f'{model.names[int(cls)]} {conf:.2f}')

    frame = cv2.resize(frame, (1080, 720))
    cv2.imshow('live feed', annotator.result())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
