
# face_landmark
目、口、眉毛の動きを出力するアプリ  

## 概要
目線の左右、口、まぶたの上下、眉毛の上下を0.0~1.0に正規化して出力します。  
また首のRPY角度をdegで出力します。  

## セットアップ
`git submodule uypdate --init`
`python3 -m venv venv`  
`. venv/bin/activate`  
`pip install -r requirements.txt`  

## 実行方法
初回はキャリブが必要

`python3 main.py --calib`  

上記コマンドで実行し、ターミナルの指示に沿ってキャリブレーションを行うと`setting/calib.json` が作成された後、アプリが実行される。  
2回目以降は下記コマンドで、このキャリブファイルを元にアプリ実行可能。  

`python3 main.py`  
