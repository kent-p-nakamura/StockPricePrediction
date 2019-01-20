README 
(後日README.mdに変換予定)

1: ソフトウェアの名前
　LearnAI (機械学習練習ソフト)

2: バージョン
　1.0

3: 配布物に関する簡単な説明
　大学の講義、卒業研究の一環として勉強している内容の一部をまとめたものです。
   v1.0は、同一データを用いたとき、入出力データや訓練データの加工処理の違いによる株価変動予測について実験することを目的に作成しました。

4: ファイル構成
	コード実行後自動的に必要なフォルダが追加で作成されます。
 StockPricePrediction (v1.0)
└── 2018
    ├── README.txt
    ├── bat_station    <---Windowsのみ使用可能
    │   ├── run_spp_main.bat.   <---ダブルクリック
    │   ├── spp_copy.py
    │   └── spp_del.py
    ├── code
    │   ├── misc_box.py
    │   ├── modify_the_data.py
    │   ├── plot_history.py
    │   ├── run_statistics.py    <---ログを用いて分析します。
    │   ├── spp_bat.py
    │   ├── spp_bat_test.py
    │   └── stock_price_prediction.py   <---これを実行します。
    └── datasets
        └──  car
                └── day
                    └── NK7203.txt <-- ここにトヨタ等の株価データをセットして学習（各自でデータの準備をお願いします。）  

5: 動作環境
	Aanaconda 3.6系 （ Windows10, macOS Mojave）
	batファイルはWindowsのみ。

6: インストールに関する情報と簡単な使用方法
	フォルダをクローン後、学習データを準備して　/2018/code/ stock_price_prediction.pyを実行します。
　使用にあたって、複数のライブラリをインポートする必要があります。未実装の場合は事前にインストールをお願いします。
　必須：
　　TensorFlow, Keras, sklearn, numpy, pandas, seaborn, matplotlib
　あれば：
　　tqdm, pydotplus, graphviz

7: 作者に関する情報
　大学生（2019.1現在）

8: 更新履歴
　2019.1.21  v1.0リリース

9:既知の不具合
　未確認

10: トラブルシューティング
	1:必要な機械学習ライブラリがインストールされていることを確かめます。
	2:学習するデータが適切なフォルダに存在するか確認します。
	3: management_flag =1 にして実行
	4: 数行ずつ実行してエラー個所を特定します。
	5; 実行環境に合わせてエラーを修正

11:謝辞
　大学にて相談に乗り、親身な指導をしていただいた教授には、お世話になりました。ここに感謝の意を表します。

12: 著作権情報とライセンス情報
The MIT License
  Copyright (c) 2018 Kenichi Nakamura
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

13: その他
    機械学習初学者として、基礎的なコーディングの流れやpythonの使い方を学習することに重点をおいて作成しました。
    参考になりましたら嬉しいです。また、コード・その他権利関係も含め、問題等ございましたら、報告頂けると幸いです。