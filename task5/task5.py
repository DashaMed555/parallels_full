import argparse
import time
import queue

import cv2
from ultralytics import YOLO
from threading import Thread, Event


skeleton = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
]


class Reader:
    def __init__(self, input_video_path, frame_queue, event_stop):
        self.cap = cv2.VideoCapture(input_video_path)
        self.frame_queue = frame_queue
        self.event_stop = event_stop

    def capture(self):
        i = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                i += 1
                self.frame_queue.put((i, frame))
            else:
                break
        self.event_stop.set()

    def __del__(self):
        self.cap.release()


class Writer:
    def __init__(self, output_video_path):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(output_video_path, fourcc, 30, (640, 480))

    def write(self, res_frames):
        for i in range(1, len(res_frames) + 1):
            self.writer.write(res_frames[i])

    def __del__(self):
        self.writer.release()


class Predictor:
    def __init__(self, frame_queue, res_frames, event_stop):
        self.frame_queue = frame_queue
        self.res_frames = res_frames
        self.event_stop = event_stop

    def predict(self):
        model = YOLO('yolov8s-pose.pt')
        while True:
            if self.frame_queue.empty():
                if self.event_stop.is_set():
                    break
            else:
                i, frame = self.frame_queue.get()
                result = model.predict(source=frame, device='cpu')[0]
                new_frame = draw_model(result)
                self.res_frames[i] = new_frame


def fun_thread_read(input_video_path, frame_queue, event_stop):
    Reader(input_video_path, frame_queue, event_stop).capture()

def fun_thread_predict(frame_queue, res_frames, event_stop):
    Predictor(frame_queue, res_frames, event_stop).predict()

def fun_thread_write(output_video_path, res_frames):
    Writer(output_video_path).write(res_frames)


def draw_model(res):
    image = res.orig_img
    for obj_num in range(res.boxes.shape[0]):
        for (start, end) in skeleton:
            x1 = int(res.keypoints.xy[obj_num, start, 0].item())
            y1 = int(res.keypoints.xy[obj_num, start, 1].item())
            x2 = int(res.keypoints.xy[obj_num, end, 0].item())
            y2 = int(res.keypoints.xy[obj_num, end, 1].item())
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                cv2.line(image,
                         (x1, y1),
                         (x2, y2),
                         (0, 255, 255))
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str)
    parser.add_argument('mode', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()

    if args.mode == 'one':
        threads_num = 1
    else:
        threads_num = 4

    threads = []
    frame_queue = queue.Queue(1000)
    res_frames = dict()
    event_stop = Event()
    thread_read = Thread(target=fun_thread_read, args=(args.video_path, frame_queue, event_stop))
    thread_read.start()

    start = time.time()
    for _ in range(threads_num):
        threads.append(Thread(target=fun_thread_predict, args=(frame_queue, res_frames, event_stop)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    thread_read.join()
    thread_write = Thread(target=fun_thread_write, args=(args.output_path, res_frames))
    thread_write.start()
    thread_write.join()
    end = time.time()

    print(f'Elapsed time: {end - start}')


if __name__ == "__main__":
    main()


# Elapsed time: 145.56425499916077  1
# Elapsed time: 130.38333106040955  2
# Elapsed time: 125.32457065582275  3
# Elapsed time: 127.88990783691406  4
