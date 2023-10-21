# https://qiita.com/mimitaro/items/bbc58051104eafc1eb38

import dlib
from imutils import face_utils
import cv2
import math
import threading
import time
import argparse
import csv

# --------------------------------
# 2.画像から顔のランドマーク検出する関数
# --------------------------------
CALIB_NUM = 10
UPR_MAX = -1.5
UPR_MIN = 0
LWR_MAX = 2.0
LWR_MIN = 0.0872665

face_detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"
face_predictor = dlib.shape_predictor(predictor_path)
lip_right_num = 49
lip_left_num = 55
mouth_upr_num = 52
mouth_lwr_num = 58
lip_right = (0, 0)
lip_left = (0, 0)
mouth_upr = (0, 0)
mouth_lwr = (0, 0)
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

    def start_capture(self):
        self.is_capturing = True
        landmark = []
        while self.is_capturing:
            ret, frame = self.camera.read()
            if ret:
                self.frame = frame
                self.frame, landmark = self.face_landmark_find(self.frame)
                if landmark is not None:
                    self.upr, self.lwr = self.calc_distance(landmark)
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

    def get_distance(self):
        with self.lock:
            return self.upr, self.lwr

    def calc_distance(self, landmark):
        for num, (x, y) in enumerate(landmark):
            if num == lip_right_num:
                lip_right = (x, y)
            if num == lip_left_num:
                lip_left = (x, y)
            if num == mouth_upr_num:
                mouth_upr = (x, y)
            if num == mouth_lwr_num:
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
    upr = (upr - UPR_MIN) / (UPR_MAX - UPR_MIN)
    lwr = (lwr - LWR_MIN) / (LWR_MAX - LWR_MIN)
    return upr, lwr


def main() -> None:
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        help="Save path",
        type=str,
    )
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
    print("口を閉じて'a'を押す")
    camera_capture.next_event.wait()
    camera_capture.next_event.clear()
    upr_close = 0
    lwr_close = 0
    # カメラ画像の表示 ('q'入力で終了)
    for i in range(0, CALIB_NUM):
        camera_capture.update_event.wait()
        camera_capture.update_event.clear()
        upr, lwr = camera_capture.get_distance()
        # 顔のランドマーク検出(2.の関数呼び出し)
        print(f"upr: {upr}, lwr: {lwr}")
        upr_close += upr
        lwr_close += lwr
    upr_close /= CALIB_NUM
    lwr_close /= CALIB_NUM

    print("口を開けて'a'を押す")
    camera_capture.next_event.wait()
    camera_capture.next_event.clear()
    upr_open = 0
    lwr_open = 0
    # カメラ画像の表示 ('q'入力で終了)
    for i in range(0, CALIB_NUM):
        camera_capture.update_event.wait()
        camera_capture.update_event.clear()
        upr, lwr = camera_capture.get_distance()
        print(f"upr: {upr}, lwr: {lwr}")
        upr_open += upr
        lwr_open += lwr
    upr_open /= CALIB_NUM
    lwr_open /= CALIB_NUM

    print(
        f"upr_close: {upr_close}, lwr_close: {lwr_close}, upr_open: {upr_open}, lwr_open: {lwr_open}"
    )

    print("'a'で記録開始")
    camera_capture.next_event.wait()
    camera_capture.next_event.clear()
    angle_vector = []
    start_time = time.time()
    # カメラ画像の表示 ('q'入力で終了)
    while not camera_capture.exit_event.is_set():
        camera_capture.update_event.wait(1)
        camera_capture.update_event.clear()
        upr, lwr = camera_capture.get_distance()
        upr_now = (upr - upr_close) / (upr_open - upr_close)
        lwr_now = (lwr - lwr_close) / (lwr_open - lwr_close)
        upr_now = filter_val(upr_now)
        lwr_now = filter_val(lwr_now)
        upr_robot, lwr_robot = convert_to_robot(upr_now, lwr_now)
        print(f"upr_robot: {upr_robot}, lwr_robot: {lwr_robot}")
        # 顔のランドマーク検出(2.の関数呼び出し)
        now = int((time.time() - start_time)* 1000)
        cur_str = f"TIMESTAMP:{now},JOINT_MOUTH_UPR:{upr_robot:.3f},JOINT_MOUTH_LWR:{lwr_robot:.3f}\n"
        angle_vector.append(cur_str)

    print("end")
    with open(args.path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(angle_vector)
    capture_thread.join()


if __name__ == "__main__":
    main()
