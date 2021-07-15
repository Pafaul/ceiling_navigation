from cv2 import cv2
from diffCalculation import diff_functions as diff_function
import time


def main():
    video_source = cv2.VideoCapture('test_video.avi')
    frame1 = None
    frame2 = None
    optical_flow_calculator = diff_function.CorrelationOpticalFlow()
    total_time = 0
    frames = 0
    while video_source.grab():
        frame2 = frame1
        flag, frame1 = video_source.retrieve()
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        if frame1 is not None:
            cv2.imshow('video1', frame1)
        if frame2 is not None:
            cv2.imshow('video2', frame2)
        cv2.waitKey(1)

        if frame1 is not None and frame2 is not None:
            start_time = time.time()
            result = optical_flow_calculator.get_optical_flow(frame1, frame2, block_size=[32, 32])
            total_time += time.time() - start_time
            frames += 1

    print(f"Per frame: {total_time/frames}")


if __name__ == '__main__':
    main()
