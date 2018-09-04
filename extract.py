from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg\yolov2.cfg", "load": "bin\yolov2.weights", "gpu": 0.9}


tfnet = TFNet(options)

imgcv = cv2.imread(".\sample_img\sample_dog.jpg")
result = tfnet.return_predict(imgcv)
print(result)

exit(0)

def video2video():
    video = "~~~.avi"  # Input video
    cap = cv2.VideoCapture(video)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    size = (width, height)
    thick = int((height + width) // 500)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # noinspection PyArgumentList
    out = cv2.VideoWriter("~~~.avi", fourcc, fps, size)  # Output video
    elapsed = int()

    while cap.isOpened():
        elapsed += 1
        ret, frame = cap.read()
        if frame is None:
            print('\nEnd of Video')
            break
        result = tfnet.return_predict(frame)
        for i in range(0, len(result), 1):
            if result[i]['confidence'] > 0.5:
                flag = False
                tl_x = result[i]['topleft']['x']
                tl_y = result[i]['topleft']['y']
                br_x = result[i]['bottomright']['x']
                br_y = result[i]['bottomright']['y']
                confidence = result[i]['confidence']
                label = result[i]['label']
                if label == 'truck':
                    box_colors = (52, 42, 212)
                    flag = True
                elif label == 'person':
                    box_colors = (128, 65, 217)
                    flag = True
                elif label == 'bus':
                    box_colors = (255, 0, 0)
                    flag = True
                elif label == 'car':
                    box_colors = (0, 198, 237)
                    flag = True
                elif label == 'motorbike':
                    box_colors = (0, 198, 14)
                    flag = True

                if flag is True:
                    cv2.rectangle(frame, (tl_x, tl_y), (br_x, br_y), box_colors, thick)
                    cv2.putText(frame, ("{0} {1:.2f}".format(label, confidence)), (tl_x, tl_y - 10), 2, 0.5, box_colors,
                                2)
        out.write(frame)
    cap.release()
    out.release()
    print("COMPLETED !!")