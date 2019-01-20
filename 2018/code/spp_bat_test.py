"""
spp_bat_test.py: spyder上で擬似バッチ処理
"""


if __name__ == '__main__':
    numM = 1
    if(numM == 0):
        print("################################################")
        print("Please use this file in other file")
        print("################################################")
    else:
        import os
        import platform
        import datetime
        from tqdm import tqdm
        #
        import stock_price_prediction as spp
        now = datetime.datetime.now()
        print("管理者用コード実行")
        time_stamp = []
        time_stamp.append(now)
        in_file = "../datasets/car/day/NK7203.txt"
        #in_file = ""
        #以下の繰り返しは、回数が多くなるとメモリ不足により学習が遅くなることがあるため、分割して学習した方が良い。
        i = 0
        N = 1
        try:
            for tq in tqdm(range(N-i+1)):
                
                spp.main(learning_cnt=i, learning_data_file_path=in_file, industry_flag=0, zero_flag=0, add_x_data_volume=0, drop_train_percent=0, 
                                     x_interval_of_days=1, y_interval_of_days=1, prediction_y_info="UpDown", management_flag=1)
                i += 1
                #gc.collect()
        except:
            print("例外がありました!")
            print("################################################")
            import traceback
            if platform.system() == "Windows":
                import winsound, time
                winsound.Beep(1000,1000)
                time.sleep(1)
                winsound.Beep(1000,1000)
                time.sleep(1)
                winsound.Beep(1000,1000)
            traceback.print_exc() 
        else:
            print("例外はありませんでした。")
        finally:
            print("メインタスクが終了しました")
            print("################################################")
        now = datetime.datetime.now()
        time_stamp.append(now)
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1000,1000)
        #print(time_stamp)
        before = time_stamp[0]
        after = time_stamp[1]
        fixTime = (after.day - before.day)*3600*24
        sumTime = fixTime + (after.hour*3600 + after.minute*60 + after.second)-(before.hour*3600 + before.minute*60 + before.second)
        hour = int(sumTime/3600)
        minute = int((sumTime%3600)/60)
        sec = int(((sumTime%3600)%60))
        
        os.remove("./temp/temp_csv.txt")
        os.remove("./temp/temp_stock_data.csv")
        print(now)
        print("{}時間{}分{}秒、掛かりました。".format(hour,minute,sec))
        print("✳︎月次以上の期間には非対応")
       


### [EOF]