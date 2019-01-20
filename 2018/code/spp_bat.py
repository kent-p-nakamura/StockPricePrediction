"""
spp_bat.py: バッチ処理時に実行するコード
"""
import platform
import datetime
import sys, os
#
import stock_price_prediction as spp


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" #Defaults to 0, from 0 to 3

args = sys.argv
i = int(args[1])
endN = int(args[2])

if __name__ == '__main__':
    now = datetime.datetime.now()
    if(i == 0): 
        print("管理者用コード実行")
        print("開始時刻：　",now)
    else: pass
    time_stamp = []
    time_stamp.append(now)
    in_file = "../datasets/car/day/NK7203.txt"
    #in_file = ""
    try:
        spp.main(learning_cnt=i, learning_data_file_path=in_file, industry_flag=0, zero_flag=0, add_x_data_volume=0, drop_train_percent=0, 
                                     x_interval_of_days=1, y_interval_of_days=1, prediction_y_info="UpDown", management_flag=1)
        '''
        spp.main(learning_cnt=i, learning_data_file_path=in_file, industry_flag=0, zero_flag=0, add_x_data_volume=0, drop_train_percent=0, 
                                     x_interval_of_days=1, y_interval_of_days=1, prediction_y_info="Weather", management_flag=1)
        '''

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
        pass
    now = datetime.datetime.now()
    time_stamp.append(now)
    if (platform.system() == "Windows") and (i == endN):
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
    print("{}回目：{}時間{}分{}秒、掛かりました。".format(i, hour,minute,sec))
    print("################################################")
    if(i == endN):
        print("終了時刻：　",now)
        
        
        
### [EOF]