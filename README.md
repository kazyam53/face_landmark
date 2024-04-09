
# face_landmark
目、口、眉毛の動きを出力するアプリ  

## 概要
上口、下口、まぶたの左右、眉毛の左右の上下を0.0~1.0に正規化して出力します。
顔のサイズや左右方向の傾きを補正しているので、実行中にある程度動いても大丈夫なはず。


## セットアップ
1. 仮想環境の作成  
`python3 -m venv venv`  
`. venv/bin/activate`  
`pip install -r requirements.txt`  

## 実行方法
`python3 facial.py`  

ターミナルの指示に沿ってキャリブを行ってください。
