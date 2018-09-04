from darkflow.net.build import TFNet
import cv2
import os
import json

DETECTION_CONFIG = {
    "model": os.path.join('.', 'cfg', 'yolov2.cfg'),
    "load": os.path.join('.', 'bin', 'yolov2.weights'),
    "gpu": 0.9,
    "threshold": 0.5
}

TARGET_CLASSES = [
    'car', 'person', 'truck', 'bus', 'motorbike'
]

TFNET = TFNet(DETECTION_CONFIG)

video_path = os.path.join('.', 'sample_video', 'video_00.mp4')
result_path = '%s.json' % video_path


def extract_detection_data_from_image(img):
    global TFNET, TARGET_CLASSES

    detection_result = TFNET.return_predict(img)

    result_dict = {}
    for class_name in TARGET_CLASSES:
        result_dict[class_name] = []

    for obj in detection_result:
        label = obj['label']
        if label not in TARGET_CLASSES:
            continue

        obj['confidence'] = float(obj['confidence'])

        obj['y_pos'] = int(obj['topleft']['y'])
        obj['x_pos'] = int(obj['topleft']['x'])
        obj['height'] = int(obj['bottomright']['y'] - obj['topleft']['y'])
        obj['width'] = int(obj['bottomright']['x'] - obj['topleft']['x'])

        result_dict[label].append(obj)

    return result_dict


def extract_from_video(video_path):
    global TARGET_CLASSES

    video_capture = cv2.VideoCapture(video_path)
    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_fps = round(video_capture.get(cv2.CAP_PROP_FPS))

    video_info = {
        'video': {
            'width': video_width,
            'height': video_height,
            'fps': float(video_fps),
            'file_name': os.path.split(video_path)[-1],
            'file_extension': video_path.split('.')[-1].lower(),
            'target_classes': TARGET_CLASSES
        },
        'frames': [

        ]
    }

    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if frame is None:
            break

        frame_data = extract_detection_data_from_image(frame)
        video_info['frames'].append(frame_data)

    return video_info


def main():
    global video_path, result_path

    with open(result_path, 'w') as fout:
        result = extract_from_video(video_path)
        json.dump(result, fout)

    print('Done!')


if __name__ == '__main__':
    main()
