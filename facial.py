# https://qiita.com/mimitaro/items/bbc58051104eafc1eb38

import dlib
from imutils import face_utils
import cv2
import math
import threading
import time
import argparse
import csv
import sounddevice as sd
import soundfile as sf
import os
import datetime

# --------------------------------
# 2.画像から顔のランドマーク検出する関数
# --------------------------------
CALIB_NUM = 10
UPR_MAX = -0.366
UPR_MIN = 0
LWR_MAX = 0.558505
LWR_MIN = 0.0872665
START_OFFSET = 0.5  # [s]
DIFF_MIN = 0.0872665

face_detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"
face_predictor = dlib.shape_predictor(predictor_path)
lip_right_num = 49
lip_left_num = 55
mouth_upr_num = 52
mouth_lwr_num = 58
eye_left_upr_num = 45
eye_left_lwr_num = 47
eye_left_left_num = 46
eye_left_right_num = 43
eye_right_upr_num = 38
eye_right_lwr_num = 42
eye_right_left_num = 40
eye_right_right_num = 37
eyeblow_left_center_num = 25
eyeblow_left_left_num = 27
eyeblow_left_right_num = 23
eyeblow_right_center_num = 20
eyeblow_right_left_num = 22
eyeblow_right_right_num = 18
lip_left_num = 55
lip_right = (0, 0)
lip_left = (0, 0)
mouth_upr = (0, 0)
mouth_lwr = (0, 0)
eye_left_upr = (0, 0)
eye_left_lwr = (0, 0)
eye_left_left = (0, 0)
eye_left_right = (0, 0)
eye_right_upr = (0, 0)
eye_right_lwr = (0, 0)
eye_right_left = (0, 0)
eye_right_right = (0, 0)
eyeblow_left_center = (0, 0)
eyeblow_left_left = (0, 0)
eyeblow_left_right = (0, 0)
eyeblow_right_center = (0, 0)
eyeblow_right_left = (0, 0)
eyeblow_right_right = (0, 0)
default_upr = 0
default_lwr = 0


# カメラのキャプチャ用のクラス
class CameraCapture:
    def __init__(self, camera_index=0):
        self.camera = cv2.VideoCapture(camera_index)
        self.frame = None
        self.lock = threading.Lock()
        self.is_capturing = False
        self.update_event = threading.Event()
        self.exit_event = threading.Event()
        self.next_event = threading.Event()
        self.left_eye_open = 0.0
        self.right_eye_open = 0.0
        self.upr = 0.0
        self.lwr = 0.0
        self.left_eyeblow = 0.0
        self.right_eyeblow = 0.0

    def start_capture(self):
        self.is_capturing = True
        landmark = []
        while self.is_capturing:
            ret, frame = self.camera.read()
            if ret:
                self.frame = frame
                self.frame, landmark = self.face_landmark_find(self.frame)
                if landmark is not None:
                    self.upr, self.lwr = self.calc_mouth_distance(landmark)
                    self.right_eye_open, self.left_eye_open = self.calc_eye_open(
                        landmark
                    )
                    self.right_eyeblow, self.left_eyeblow = self.calc_eyeblow_distance(
                        landmark
                    )
                self.update_event.set()
            # 結果の表示
            cv2.imshow("img", self.frame)
            # 'q'が入力されるまでループ
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                # 後処理
                self.exit_event.set()
                self.camera.release()
                cv2.destroyAllWindows()
                break
            elif key & 0xFF == ord("a"):
                self.next_event.set()

    def stop_capture(self):
        self.is_capturing = False
        self.camera.release()

    def get_frame(self):
        return self.frame

    def get_mouth_distance(self):
        with self.lock:
            return self.upr, self.lwr

    def calc_mouth_distance(self, landmark):
        for num, (x, y) in enumerate(landmark):
            if num == lip_right_num:
                lip_right = (x, y)
            if num == lip_left_num:
                lip_left = (x, y)
            if num == mouth_upr_num:
                mouth_upr = (x, y)
            if num == mouth_lwr_num:
                mouth_lwr = (x, y)
            if num == eye_left_upr_num:
                mouth_lwr = (x, y)
            if num == eye_left_lwr_num:
                mouth_lwr = (x, y)
            if num == eye_right_upr_num:
                mouth_lwr = (x, y)
            if num == eye_right_lwr_num:
                mouth_lwr = (x, y)
        distance_upr = abs(
            (lip_left[0] - lip_right[0]) * (lip_right[1] - mouth_upr[1])
            - (lip_right[0] - mouth_upr[0]) * (lip_left[1] - lip_right[1])
        ) / math.sqrt(
            (lip_left[0] - lip_right[0]) ** 2 + (lip_left[1] - lip_right[1]) ** 2
        )
        distance_lwr = abs(
            (lip_left[0] - lip_right[0]) * (lip_right[1] - mouth_lwr[1])
            - (lip_right[0] - mouth_lwr[0]) * (lip_left[1] - lip_right[1])
        ) / math.sqrt(
            (lip_left[0] - lip_right[0]) ** 2 + (lip_left[1] - lip_right[1]) ** 2
        )
        return distance_upr, distance_lwr

    def get_eye_open(self):
        with self.lock:
            return self.left_eye_open, self.right_eye_open

    def calc_eye_open(self, landmark):
        for num, (x, y) in enumerate(landmark):
            if num == eye_left_upr_num:
                eye_left_upr = (x, y)
            if num == eye_left_lwr_num:
                eye_left_lwr = (x, y)
            if num == eye_left_left_num:
                eye_left_left = (x, y)
            if num == eye_left_right_num:
                eye_left_right = (x, y)
            if num == eye_right_upr_num:
                eye_right_upr = (x, y)
            if num == eye_right_lwr_num:
                eye_right_upr = (x, y)
            if num == eye_right_left_num:
                eye_right_left = (x, y)
            if num == eye_right_right_num:
                eye_right_right = (x, y)
        left_eye_open = abs(
            (eye_left_left[0] - eye_left_right[0])
            * (eye_left_right[1] - (eye_left_upr[1] - eye_left_lwr[1]))
            - (eye_left_right[0] - eye_left_upr[0])
            * (eye_left_left[1] - eye_left_right[1])
        ) / math.sqrt(
            (eye_left_left[0] - eye_left_right[0]) ** 2
            + (eye_left_left[1] - eye_left_right[1]) ** 2
        )
        right_eye_open = abs(
            (eye_right_left[0] - eye_right_right[0])
            * (eye_right_right[1] - (eye_right_upr[1] - eye_right_lwr[1]))
            - (eye_right_right[0] - eye_right_lwr[0])
            * (eye_right_left[1] - eye_right_right[1])
        ) / math.sqrt(
            (eye_right_left[0] - eye_right_right[0]) ** 2
            + (eye_right_left[1] - eye_right_right[1]) ** 2
        )
        return left_eye_open, right_eye_open

    def get_eyeblow(self):
        with self.lock:
            return self.left_eyeblow, self.right_eyeblow

    def calc_eyeblow_distance(self, landmark):
        for num, (x, y) in enumerate(landmark):
            if num == eyeblow_left_center_num:
                eyeblow_left_center = (x, y)
            if num == eye_left_left_num:
                eye_left_left = (x, y)
            if num == eye_left_right_num:
                eye_left_right = (x, y)
            if num == eyeblow_right_center_num:
                eyeblow_right_center = (x, y)
            if num == eye_right_left_num:
                eye_right_left = (x, y)
            if num == eye_right_right_num:
                eye_right_right = (x, y)
        eyeblow_left = abs(
            (eye_left_left[0] - eye_left_right[0])
            * (eye_left_right[1] - eyeblow_left_center[1])
            - (eye_left_right[0] - mouth_upr[0])
            * (eye_left_left[1] - eye_left_right[1])
        ) / math.sqrt(
            (eye_left_left[0] - eye_left_right[0]) ** 2
            + (eye_left_left[1] - eye_left_right[1]) ** 2
        )
        eyeblow_right = abs(
            (eye_right_left[0] - eye_right_right[0])
            * (eye_right_right[1] - eyeblow_right_center[1])
            - (eye_right_right[0] - mouth_upr[0])
            * (eye_right_left[1] - eye_right_right[1])
        ) / math.sqrt(
            (eye_right_left[0] - eye_right_right[0]) ** 2
            + (eye_right_left[1] - eye_right_right[1]) ** 2
        )
        return eyeblow_left, eyeblow_right

    def face_landmark_find(self, img):
        # 顔検出
        img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector(img_gry, 1)
        landmark = None
        # 検出した全顔に対して処理
        for face in faces:
            # 顔のランドマーク検出
            landmark = face_predictor(img_gry, face)
            # 処理高速化のためランドマーク群をNumPy配列に変換(必須)
            landmark = face_utils.shape_to_np(landmark)

            # ランドマーク描画
            for num, (x, y) in enumerate(landmark):
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

        return img, landmark


# カメラのキャプチャを別スレッドで実行
def capture_frames(camera_capture):
    camera_capture.start_capture()


def filter_val(input):
    if input > 1.0:
        input = 1.0
    elif input < 0.0:
        input = 0.0
    return input


def convert_to_robot(upr, lwr):
    upr = upr * (UPR_MAX - UPR_MIN) + UPR_MIN
    lwr = lwr * (LWR_MAX - LWR_MIN) + LWR_MIN
    # if lwr - upr < DIFF_MIN:
    #    lwr = upr + DIFF_MIN
    return upr, lwr


def main() -> None:
    # parse arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # --------------------------------
    # 1.顔ランドマーク検出の前準備
    # --------------------------------
    # 顔ランドマーク検出ツールの呼び出し

    # --------------------------------
    # 3.カメラ画像の取得
    # --------------------------------
    # カメラの指定(適切な引数を渡す)
    camera_capture = CameraCapture()
    capture_thread = threading.Thread(target=capture_frames, args=(camera_capture,))
    capture_thread.start()
    time.sleep(2)

    print("キャリブ1: 口と目を閉じて画像ウィンドウ上で'a'を押す")
    camera_capture.next_event.wait()
    camera_capture.next_event.clear()
    upr_close = 0
    lwr_close = 0
    left_eye_close = 0
    right_eye_close = 0
    left_eyeblow_close = 0
    right_eyeblow_close = 0
    # カメラ画像の表示 ('q'入力で終了)
    for i in range(0, CALIB_NUM):
        camera_capture.update_event.wait()
        camera_capture.update_event.clear()
        upr, lwr = camera_capture.get_mouth_distance()
        left_eye, right_eye = camera_capture.get_eye_open()
        left_eyeblow, right_eyeblow = camera_capture.get_eyeblow()
        # 顔のランドマーク検出(2.の関数呼び出し)
        print(
            f"upr: {upr}, lwr: {lwr}, left_eye: {left_eye}, right_eye: {right_eye}, left_eyeblow: {left_eyeblow}, right_eyeblow: {right_eyeblow}"
        )
        upr_close += upr
        lwr_close += lwr
        left_eye_close += left_eye
        right_eye_close += right_eye
        left_eyeblow_close += left_eyeblow
        right_eyeblow_close += right_eyeblow
    upr_close /= CALIB_NUM
    lwr_close /= CALIB_NUM
    left_eye_close /= CALIB_NUM
    right_eye_close /= CALIB_NUM
    left_eyeblow_close /= CALIB_NUM
    right_eyeblow_close /= CALIB_NUM

    print("キャリブ2: 口と目をなるべく開けて画像ウィンドウ上で'a'を押す")
    camera_capture.next_event.wait()
    camera_capture.next_event.clear()
    upr_open = 0
    lwr_open = 0
    left_eye_open = 0
    right_eye_open = 0
    left_eyeblow_open = 0
    right_eyeblow_open = 0
    # カメラ画像の表示 ('q'入力で終了)
    for i in range(0, CALIB_NUM):
        camera_capture.update_event.wait()
        camera_capture.update_event.clear()
        upr, lwr = camera_capture.get_mouth_distance()
        left_eye, right_eye = camera_capture.get_eye_open()
        left_eyeblow, right_eyeblow = camera_capture.get_eyeblow()
        print(
            f"upr: {upr}, lwr: {lwr}, left_eye: {left_eye}, right_eye: {right_eye}, left_eyeblow: {left_eyeblow}, right_eyeblow: {right_eyeblow}"
        )
        upr_open += upr
        lwr_open += lwr
        left_eye_open += left_eye
        right_eye_open += right_eye
        left_eyeblow_open += left_eyeblow
        right_eyeblow_open += right_eyeblow
    upr_open /= CALIB_NUM
    lwr_open /= CALIB_NUM
    left_eye_open /= CALIB_NUM
    right_eye_open /= CALIB_NUM
    left_eyeblow_open /= CALIB_NUM
    right_eyeblow_open /= CALIB_NUM

    print(
        f"upr_close: {upr_close}, lwr_close: {lwr_close}, upr_open: {upr_open}, lwr_open: {lwr_open}"
    )

    print("画像ウィンドウ上で'a'で計測開始。'q'で終了")
    camera_capture.next_event.wait()
    camera_capture.next_event.clear()
    angle_vector = []
    start_time = time.time()
    # カメラ画像の表示 ('q'入力で終了)
    while not camera_capture.exit_event.is_set():
        cur_str = []
        camera_capture.update_event.wait(1)
        camera_capture.update_event.clear()
        # 口の開き幅を計算
        upr, lwr = camera_capture.get_mouth_distance()
        upr_now = (upr - upr_close) / (upr_open - upr_close)
        lwr_now = (lwr - lwr_close) / (lwr_open - lwr_close)
        upr_now = filter_val(upr_now)
        lwr_now = filter_val(lwr_now)
        # 目の開き幅を計算
        left_eye, right_eye = camera_capture.get_eye_open()
        left_eye_now = (left_eye - left_eye_close) / (left_eye_open - left_eye_close)
        right_eye_now = (right_eye - right_eye_close) / (
            right_eye_open - right_eye_close
        )
        left_eye_now = filter_val(left_eye_now)
        right_eye_now = filter_val(right_eye_now)
        # 眉毛の上下を計算
        left_eyeblow, right_eyeblow = camera_capture.get_eyeblow()
        left_eyeblow_now = (left_eyeblow - left_eyeblow_close) / (
            left_eyeblow_open - left_eyeblow_close
        )
        right_eyeblow_now = (right_eyeblow - right_eyeblow_close) / (
            right_eyeblow_open - right_eyeblow_close
        )
        left_eyeblow_now = filter_val(left_eyeblow_now)
        right_eyeblow_now = filter_val(right_eyeblow_now)

        # 計算結果を表示(0.0~1.0に正規化)
        print(f"upr: {upr_now}, lwr: {lwr_now}")
        print(f"left_eye: {left_eye_now}, right_eye: {right_eye_now}")
        print(f"left_eyeblow: {left_eyeblow_now}, right_eyeblow: {right_eyeblow_now}")

        # 顔のランドマーク検出(2.の関数呼び出し)
        now = int((time.time() - start_time - START_OFFSET) * 1000)
        if now <= 0.0:
            continue
    # 最後の1つを消す
    print("end")
    capture_thread.join()


if __name__ == "__main__":
    main()
