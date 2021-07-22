from cv2 import cv2
from diffCalculation import diff_functions as diff_function
import time


def main():
    video_source = cv2.VideoCapture('test_video.avi')
    frame1 = None
    frame2 = None
    optical_flow_calculator = diff_function.CorrelationOpticalFlow()
    frames = 0
    times = []
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
            result = optical_flow_calculator.get_optical_flow(frame1, frame2, block_size=[32, 32], search_window_size=[1, 1])
            # optical_flow_calculator.filter_optical_flow(result)
            times.append(time.time() - start_time)
            image = optical_flow_calculator.draw_optical_flow(frame2, result)
            cv2.imshow('optical flow', cv2.resize(image, (image.shape[1]*2, image.shape[0]*2)))
            cv2.waitKey(0)

            frames += 1

    print(f"Per frame: {sum(times)/frames}")
    print(f"Maximum time: {max(times)}")
    print(f"Minimum time: {min(times)}")
    print(f"Total frames: {len(times)}")


if __name__ == '__main__':
    main()
