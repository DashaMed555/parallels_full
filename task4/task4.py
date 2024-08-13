import time
import logging
import queue
import argparse
import threading

import cv2


class Sensor:
    def get(self):
        raise NotImplementedError('Subclasses must implement method get()')


class SensorX(Sensor):
    """Sensor X"""
    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data


class SensorCam(Sensor):
    def __init__(self, camera_name, resolution):
        try:
            self.camera = cv2.VideoCapture(camera_name)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        except cv2.error:
            logging.error(f'Could not find camera {camera_name}')
            exit(1)

    def get(self):
        ret, frame = self.camera.read()
        if not ret:
            logging.error(f'Could not read frame')
        return frame

    def __del__(self):
        self.camera.release()


class WindowImage:
    def __init__(self, fps):
        self._fps = fps

    def show(self, image):
        cv2.imshow('image', image)
        time.sleep(1 / self._fps)

    def __del__(self):
        cv2.destroyAllWindows()


def thread_worker(sensor, q):
    while True:
        data = sensor.get()
        q.put(data)


def create_thread(sensor, q):
    thr = threading.Thread(target=thread_worker, args=(sensor, q))
    thr.daemon = True
    return thr


def main():
    logging.basicConfig(filename='log/task.log')

    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_name', type=str)
    parser.add_argument('--resolution', type=str)
    parser.add_argument('--fps', type=int)
    args = parser.parse_args()

    camera_name = int(args.camera_name)
    resolution = tuple(map(int, args.resolution.split('x')))
    fps = args.fps

    sensor0 = SensorX(0.01)
    sensor1 = SensorX(0.1)
    sensor2 = SensorX(1)

    q0 = queue.Queue()
    q1 = queue.Queue()
    q2 = queue.Queue()

    thr0 = create_thread(sensor0, q0)
    thr1 = create_thread(sensor1, q1)
    thr2 = create_thread(sensor2, q2)

    threads = [thr0, thr1, thr2]
    for thread in threads:
        thread.start()

    window = WindowImage(fps)
    camera = SensorCam(camera_name, resolution)

    sensor0_value = 0
    sensor1_value = 0
    sensor2_value = 0

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if not q0.empty():
            sensor0_value = q0.get()

        if not q1.empty():
            sensor1_value = q1.get()

        if not q2.empty():
            sensor2_value = q2.get()

        image = camera.get()
        if image is not None:
            h, w, c = image.shape
            cv2.putText(image, f'Sensor0: {sensor0_value}',
                        (w - 120, h - 70),
                        1,
                        cv2.FONT_HERSHEY_PLAIN,
                        (255, 255, 255), 2)

            cv2.putText(image, f'Sensor1: {sensor1_value}',
                        (w - 120, h - 50),
                        1,
                        cv2.FONT_HERSHEY_PLAIN,
                        (255, 255, 255), 2)

            cv2.putText(image, f'Sensor2: {sensor2_value}',
                        (w - 120, h - 30),
                        1,
                        cv2.FONT_HERSHEY_PLAIN,
                        (255, 255, 255), 2)
            window.show(image)


if __name__ == '__main__':
    main()
