import cv2

# Read video
def read_video(filename):
    cap = cv2.VideoCapture(filename)
    video_name = filename.split('/')[-1].split('.')[0]
    video = cv2.VideoCapture(filename)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    return (video_name, video, frame_width, frame_height)

# Read first frame.
def read_frame(video):
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
    return (ok, frame)

# Load a model imported from Tensorflow'
def load_model():
    classNames = {0: 'background',
                  1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
                  7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
                  13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
                  18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
                  24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
                  32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
                  37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
                  41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
                  46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                  51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                  56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                  61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
                  67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
                  75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
                  80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
                  86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

    pbtxt="ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
    pbgraph = "frozen_inference_graph.pb"
    tensorflowNet = cv2.dnn.readNetFromTensorflow(pbgraph, pbtxt)
    return (tensorflowNet, classNames)

# Use the given image as input, which needs to be blob(s).
def run_model(tensorflowNet, classNames, frame, frame_width, frame_height):
    tensorflowNet.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False))

    # Runs a forward pass to compute the net output
    networkOutput = tensorflowNet.forward()

    # Loop on the outputs
    bboxlist = []
    for detection in networkOutput[0, 0]:
        score = float(detection[2])
        if score > 0.5:
            desc = classNames[int(detection[1])]
            left = detection[3] * frame_width
            top = detection[4] * frame_height
            right = detection[5] * frame_width
            bottom = detection[6] * frame_height
            bboxlist.append((desc, score, left, top, right, bottom))

    ok = len(bboxlist) > 0
    return (ok, bboxlist)

def draw_bbox(frame, bboxlist, fps=False):
    for bbox in bboxlist:
        # draw a red rectangle around detected objects
        desc, score, left, top, right, bottom = bbox
        text = f"{desc}: {int(score*100)}%"
        fontScale = cv2.getFontScaleFromHeight(fontFace=cv2.FONT_HERSHEY_COMPLEX, pixelHeight=10, thickness=1)
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, fontScale=fontScale, thickness=1)[0]

        #labels
        cv2.rectangle(frame, (int(left), int(top-6-text_height)), (int(left+text_width), int(top)), (0,0,255), cv2.FILLED)
        cv2.putText(frame, text, (int(left), int(top-5)), cv2.FONT_HERSHEY_COMPLEX, fontScale, (255, 255, 255), 1, cv2.LINE_AA);

        #bbox
        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)

        if fps:
            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA);

    return frame

def run_detector(filename="videos/cycle.mp4", save_video=False):
    video_name, video, frame_width, frame_height = read_video(filename)
    tensorflowNet, classNames = load_model()

    if save_video:
        out = cv2.VideoWriter('{}_bbox.mp4'.format(video_name), cv2.VideoWriter_fourcc(*'MP4V'),
                              30, (640, 360))

    while True:
        # Read a new frame
        ok, frame = read_frame(video)
        if not ok: break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bboxlist = run_model(tensorflowNet, classNames, frame, frame_width, frame_height)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            # Tracking success
            frame = draw_bbox(frame, bboxlist, fps)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2, cv2.LINE_AA)

        # Display result
        cv2.imshow("Detector", frame)

        if save_video:
            outframe = cv2.resize(frame, (640,360))
            out.write(outframe)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

    if save_video:
        out.release()

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detector(save_video=True)
