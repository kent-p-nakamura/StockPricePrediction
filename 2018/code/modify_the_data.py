"""
modify_the_data.py: データ加工用モジュール
"""
import shutil
import pandas as pd
import numpy as np
import collections
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
#
import misc_box


#関係するフォルダ
folders = {
            "temp": "./temp/"
            }


def csv_to_df(input_file_path, input_columns_info="Basic", output_y_info="All", 
                NK255_flag=0, x_interval_of_days=1, y_interval_of_days=1, zero_flag=0, management_flag=0):
    """
    引数の説明
    input_file_path: 加工対象のデータファイル(txt, csv)パス
    
    input_columns_info: {
                        "Basic": ["Date", "Open", "High", Low", "Close", "Vol"],
                        "Close": ["Date", "Open", "High", Low", "Close", "Vol"]
                        }

    output_y_info:      {
                        "All": ["Weather", "UpDown"],
                        "UpDown": ["UpDown"],
                        "Weather": ["Weather"]
                        }
    
    
    NK255_flag:  日経平均株価情報を使用するかの有無
    
    x_interval_of_days: 説明変数Xの日数差
    y_interval_of_days: 目的変数Yの日数差
    management_flagF:　デバッグ用のコメント表示の有無
    """
            
    #input_file_pathの「型」と「拡張子」をテェック
    file_extension_check = type_and_extension_check(input_file_path)
    if(management_flag == 1): print("line49: file_extension_check: "+str(file_extension_check))
    #使用データ準備（リストで管理）
    if (file_extension_check == False):
        #使用するファイルのpathを外部から入力後、リスト化
        out_string = "加工対象のファイル(.txt or .csv)"
        while file_extension_check != True:
            path_list = misc_box.input_file_list(out_string)
            file_extension_check = type_and_extension_check(path_list) 
    elif(file_extension_check == True):
        temp_list = []
        path_type = type(input_file_path)
        if(path_type == str):
            temp_list.append(input_file_path)
        elif(path_type == list):
            for temp_file in input_file_path:
                temp_list.append(temp_file)
        else: pass
        path_list = temp_list
        del temp_list
    else: pass
    
    
    #使用するファイル名を表示
    misc_box.print_file_name(path_list)
    if(management_flag == 1): print("line73: 加工対象データファイルのpath情報取得")
    
    
    #加工対象データの正規化（txtファイルなら、csvに変換）後に、tempファイルの作成
    temp_cnt = 1
    temp_list = []
    for temp_file in path_list:
        out_csv_path = folders["temp"]+"temp_scv_{}.csv".format(temp_cnt)
        shutil.copyfile(temp_file, out_csv_path)
        temp_list.append(out_csv_path)
        temp_cnt += 1
    path_list = temp_list
    del temp_list
    if(management_flag == 1): print("line86: csv保存完了")
    #再利用できるように、pathを保存
    with open("./temp/temp_csv_path.txt", mode='w') as f:
        f.write('\n'.join(path_list))
    
    
    #学習データ(X_data, Y_data)作成
    if(input_columns_info == "Basic"): #["Date", "Open", "High", Low", "Close", "Vol"]使用のとき
        if(management_flag == 1): print("line98: Columns: Basicで処理開始")
        #使用するcsvファイルのpathを読み込み
        with open(folders["temp"]+"temp_csv_path.txt", 'r') as f:
            temp_csv_path_list = [s.strip() for s in f.readlines()]
        csv_path = temp_csv_path_list[0]
        del temp_csv_path_list
        
        #csvからDataFrameを読み込み
        input_original_df = pd.read_csv(csv_path, encoding='utf-8')
        temp_x_df = input_original_df.loc[:, ["Open", "High", "Low", "Close", "Vol"]]
        temp_Date_df = input_original_df.Date
        temp_rClose_df = input_original_df.Close
        
        x_now = temp_x_df #Xデータ用
        y_now = temp_x_df #Yデータ用
        
        #Xデータ準備
        x_before = x_now.shift(int(x_interval_of_days))
        x_df = (x_now - x_before)/x_before
        
        out_df = x_df
        out_df["Date"] = temp_Date_df
        out_df["rClose"] = temp_rClose_df
        del temp_Date_df
        
        #Yデータ準備
        y_after = y_now.shift(-int(y_interval_of_days))
        y_df = (y_after - y_now) / y_now
        benefit = temp_rClose_df.shift(-int(y_interval_of_days)) - temp_rClose_df
        out_df["Benefit"] = benefit
        del temp_rClose_df
        
        
        #四分位数を利用してラベルを作成
        Q1 = float(y_df.Close.quantile(.25)) #第1四分位数
        Q2 = float(y_df.Close.quantile(.50)) #第2四分位数
        Q3 = float(y_df.Close.quantile(.75)) #第3四分位数
        #Weatherのデータを作成
        if(management_flag == 0): time.sleep(0.5)
        if(management_flag == 1): print("line133: Weatherデータ準備"); time.sleep(0.5)
        temp_list = []
        cnt=0
        for temp_Close in tqdm(y_df.Close, desc="Weather: "):
            if(temp_Close < Q1):
                temp_list.append(0)
            elif(temp_Close < Q2):
                temp_list.append(1)
            elif(temp_Close == Q2):  #
                if(zero_flag == 0):
                    temp_list.append(1)
                elif(zero_flag == 1):
                    temp_list.append(4) #
                elif(zero_flag == 2):
                    if(cnt%2==0):
                        temp_list.append(1)
                    else:
                        temp_list.append(2)
                    cnt+=1
                else:
                    print("########################################")
                    print("")
                    print("zero_flag　のエラーです。")
                    print("")
                    print("########################################")
            elif(temp_Close < Q3):
                temp_list.append(2)
            elif(temp_Close >= Q3):
                temp_list.append(3)
            else: temp_list.append(None)
        out_df["Weather"] = temp_list
        del temp_list
        
        #UpDownのデータを作成
        if(management_flag == 1): print("line164: UpDownデータ準備"); time.sleep(0.5)
        temp_list = []
        cnt=0
        for temp_Close in tqdm(y_df.Close, desc="UpDown: "):
            if(temp_Close < 0.0):
                temp_list.append(0)
            elif(temp_Close == 0.0):
                if(zero_flag == 0):
                    temp_list.append(0)
                elif(zero_flag == 1):
                    temp_list.append(2)
                elif(zero_flag == 2):
                    if(cnt%2==0):
                        temp_list.append(0)
                    else:
                        temp_list.append(1)
                    cnt+=1
                else:
                    print("########################################")
                    print("")
                    print("zero_flag　のエラーです。")
                    print("")
                    print("########################################")
            elif(temp_Close > 0.0):
                temp_list.append(1)
            else: temp_list.append(None)        
        out_df["UpDown"] = temp_list
        del temp_list
        
        out_df.dropna(how="any", inplace=True)
        
        if(output_y_info == "All"):
            out_df = out_df.loc[:, ["Date", "Open", "High", "Low", "Close", "Vol", "rClose", "UpDown", "Weather", "Benefit"]]
        elif(output_y_info == "UpDown"):
            out_df = out_df.loc[:, ["Date", "Open", "High", "Low", "Close", "Vol", "rClose", "UpDown", "Benefit"]]
        elif(output_y_info == "Weather"):
            out_df = out_df.loc[:, ["Date", "Open", "High", "Low", "Close", "Vol", "rClose", "Weather", "Benefit"]]
        else:
            print("output_y_infoは、")
            print("Weather ⇐ [\"Weather\"]出力")
            print("UpDown ⇐ [\"UpDown\"]出力")
            print("All ⇐ [\"Weather\", \"UpDown\"]出力")
            print("のいずれかを入力してください。")
    
    #下記、準備中。
    elif(input_columns_info == "Industry"): #多数ファイルで["Date", "Close", "Close1", ..., "CloseN"]使用
        """
        if(management_flag == 1): print("◎多数ファイル処理開始")
        #Closeデータ集csvを生成
        if(NK255_flag == 0): 
            NK255_flag = temp_csv[0]
            temp_csv = temp_csv[1:]
        
        out_data = []
        out_data = pd.read_csv(NK255_flag, encoding='utf-8')
        out_data = out_data.loc[:, ["Date", "Close"]]
    
        num = 1
        for j in temp_csv:
            temp_data = pd.read_csv(j, encoding='utf-8')
            temp_data = temp_data.loc[:, ["Date", "Close"]]
    
    
            if(out_data.iloc[0, 0] > temp_data.iloc[0, 0]):
                #print(out_data.iloc[0, 0], "が大きい")
                while (out_data.iloc[0, 0] != temp_data.iloc[0, 0]):
                    temp_data = temp_data.shift(-1)
                out_data["Close{}".format(num)] = temp_data.loc[:,["Close"]]
            else:
                #print(out_data.iloc[0, 0], "が小さい")
                while (out_data.iloc[0, 0] != temp_data.iloc[0, 0]):
                    out_data = out_data.shift(-1)
                out_data["Close{}".format(num)] = temp_data.loc[:,["Close"]]
            num += 1
            
        
        out_data = out_data.dropna(how='any')
        #print(out_data)
        if(management_flag == 1): print("Closeデータ集csv生成完了")
        
        
        #学習データ(X_data, Y_data)作成
        temp_data = out_data
        out_data = []
        temp_data2 = temp_data.loc[:, temp_data.columns != "Date"]
        temp_Date = temp_data.loc[:, ["Date"]]
        
        temp1_now = temp_data2 #Xデータ用
        
        temp1_before = temp1_now.shift(int(x_interval_of_days))
        X_data = (temp1_now - temp1_before)/temp1_before
        out_temp = X_data
        out_temp["Date"] = temp_Date
        
        ex_data = out_temp.shift(-int(x_interval_of_days))
        #0より大きいサンプルの数
        tempTF = (ex_data >= 0)
        tempTF = tempTF.sum(axis=1)
        
        #列の数で天気を予想
        col = len(out_temp.columns)-1
        
        
        i = 0
        temp_list = []
        while i < len(tempTF):
            if((tempTF.iloc[i]-1)<=col*(1/4)):
                temp_list.append("Rainy")
            elif((tempTF.iloc[i]-1)<=col*(2/4)):
                temp_list.append("CloudyL")
            elif((tempTF.iloc[i]-1)<=col*(3/4)):
                temp_list.append("CloudyH")
            else:
                temp_list.append("Sunny")
            i += 1
        out_temp["Weather"] = temp_list
        
        
        i = 0
        temp_list = []
        while i < len(tempTF):
            if((tempTF.iloc[i]-1)<=col*(1/2)):
                temp_list.append(0) #Downの場合
            else:
                temp_list.append(1) #Upの場合
            i += 1
        out_temp["UpDown"] = temp_list
        
        out_temp = out_temp.dropna(how='any')
        
        
        if(output_y_info == 0):
            out_data = out_temp.loc[:, out_temp.columns != "UpDown"]
        elif(output_y_info == 1):
            out_data = out_temp.loc[:, out_temp.columns != "Weather"]
        elif(output_y_info == 2):
            out_data = out_temp.loc[:, :]
        elif(output_y_info == 3):
            out_data = out_temp.loc[:, out_temp.columns != "UpDown"]
            out_data = out_data.loc[:, out_temp.columns != "Weather"]
        else:
            print("output_y_infoは、")
            print("0 ⇐ [\"Weather\"]出力")
            print("1 ⇐ [\"UpDown\"]出力")
            print("2 ⇐ [\"Weather\", \"UpDown\"]出力")
            print("のいずれかを入力してください。")
        if(management_flag == 1): print("出力データ準備完了")
        """
        pass
    else:
        print("input_columns_infoは、")
        print("Basic ⇐ １つのファイル入力の場合")
        print("Industry ⇐ 多数ファイル（リスト）の場合")
        print("のいずれかを入力してください。")
    if(management_flag == 1): out_df.to_csv(folders["temp"]+"temp_stock_data.csv")
    return out_df

def drop_train_group(x_train_group, y_train_group, drop_train_percent):
    y_columns_index =  len(x_train_group[0])
    temp_drop_df = pd.DataFrame(x_train_group)
    temp_drop_df[y_columns_index] =  y_train_group[:, 0] #UpDown
    temp_drop_df[y_columns_index+1] =  y_train_group[:, 1] #Weather
    temp_drop_df[y_columns_index+2] =  y_train_group[:, 2] #Benefit
    temp_drop_df["CloseT"] = abs(temp_drop_df.loc[:, 3]) #Close
    
    
    temp_drop_df = temp_drop_df.sort_values("CloseT", ascending=True)
    try:
        drop_amounts = int(len(temp_drop_df)*(drop_train_percent*0.01))
    except ValueError : #ValueErrorの場合の処理
        print("drop_train_percentには、0以上100未満の数値を入れてください。")
        print("drop_train_percent: ",drop_train_percent)
    
    temp_drop_df = temp_drop_df.shift(-int(drop_amounts))
    temp_drop_df = temp_drop_df.dropna(how='any')
    temp_drop_df = temp_drop_df.loc[:, temp_drop_df.columns != "CloseT"]
    
    x_train_group = temp_drop_df.loc[:, temp_drop_df.columns != y_columns_index+2]
    x_train_group = x_train_group.loc[:, x_train_group.columns != y_columns_index+1]
    x_train_group = x_train_group.loc[:, x_train_group.columns != y_columns_index]
    x_train_group = np.array(x_train_group)
    y_train_group = np.array(temp_drop_df.loc[:, [y_columns_index,y_columns_index+1,y_columns_index+2]])
    del temp_drop_df
    return x_train_group, y_train_group



def calculate_df(original_df, x_interval_of_time, management_flag=0):
    '''
    引数の説明
    input_columns_info: inputフラグ   0⇒1ファイルで["Date", "Open", "High", Low", "Close", "Vol"]使用
                        1⇒多数ファイルで["Date", "Close", "Close1", ..., "CloseN"]使用
    output_y_info: outputフラグ 0⇒["Weather"]出力
                        1⇒["UpDown"]出力
                        2⇒["Weather", "UpDown"]出力
    NK255_flag:  日経２５５      0⇒日経平均株価情報を使用しない
                       path⇐ファイルのパスを入力
    original_df:           path⇐データファイル(txt, csv)のパスを入力,リストにて複数可 
    x_interval_of_time: N日前からの増加減少率
    y_interval_of_days: N日後の予測増加減少率
    management_flagF:    管理者フラグ 0⇒pass
                          1⇒コンソールに途中経過出力
    '''
    
    #DataFrameを読み込み
    temp_df = original_df
    temp_x_df = temp_df.loc[:, ["Open", "High", "Low", "Close", "Vol"]]
    temp_Date_df = temp_df.loc[:, ["Date"]]
    temp_Time_df = temp_df.loc[:, ["Time"]]
    temp_rClose_df = temp_df.loc[:, ["Close"]] 
    
    x_now = temp_x_df #Xデータ用
    
    #Xデータ準備
    x_before = x_now.shift(int(x_interval_of_time))
    x_df = (x_now - x_before)/x_before
    
    out_df = x_df
    out_df["Date"] = temp_Date_df
    out_df["Time"] = temp_Time_df
    out_df["rClose"] = temp_rClose_df
    del temp_Date_df



    out_df = out_df.fillna(0)
    out_df = out_df.loc[:, ["Date","Time", "Open", "High", "Low", "Close", "Vol", "rClose"]]   
    
    return out_df



def make_cnn_data(input_file_path, missing_values_range=0, management_flag=0):
    
    #input_file_pathの「型」と「拡張子」をテェック
    file_extension_check = type_and_extension_check(input_file_path)
    if(management_flag == 1): print("line403: file_extension_check: "+str(file_extension_check))
    #使用データ準備（リストで管理）
    if (file_extension_check == False):
        #使用するファイルのpathを外部から入力後、リスト化
        out_string = "10minの加工対象のファイル(.txt or .csv)"
        while file_extension_check != True:
            path_list = misc_box.input_file_list(out_string)
            file_extension_check = type_and_extension_check(path_list) 
    elif(file_extension_check == True):
        temp_list = []
        path_type = type(input_file_path)
        if(path_type == str):
            temp_list.append(input_file_path)
        elif(path_type == list):
            for temp_file in input_file_path:
                temp_list.append(temp_file)
        else: pass
        path_list = temp_list
        del temp_list
    else: pass
    
    
    #使用するファイル名を表示
    misc_box.print_file_name(path_list)
    if(management_flag == 1): print("line427: 加工対象データファイルのpath情報取得!")

    #加工対象データの正規化（txtファイルなら、csvに変換）後に、tempファイルの作成
    temp_cnt = 1
    temp_list = []
    temp_list2 = []
    for temp_file in path_list:
        out_csv_path = folders["temp"]+"temp_scv_{}.csv".format(temp_cnt)
        shutil.copyfile(temp_file, out_csv_path)
        stock_code = re.findall('NK[0-9]{4}' , temp_file)
        temp_list.append(out_csv_path)
        temp_list2.append(stock_code[0])
        temp_cnt += 1
    path_list = temp_list
    stock_code_list = temp_list2
    del temp_list, temp_list2
    if(management_flag == 1): print("line493: csv保存完了")
    #再利用できるように、pathを保存
    with open("./temp/temp_csv_path.txt", mode='w') as f:
        f.write('\n'.join(path_list))
    with open("./temp/temp_stock_code_path.txt", mode='w') as f:
        f.write('\n'.join(stock_code_list))


    #使用するcsvファイルのpathを読み込み
    with open(folders["temp"]+"temp_csv_path.txt", 'r') as f:
        temp_csv_path_list = [s.strip() for s in f.readlines()]
    csv_path = temp_csv_path_list[0]
    del temp_csv_path_list
    with open(folders["temp"]+"temp_stock_code_path.txt", 'r') as f:
        temp_stock_code_list = [s.strip() for s in f.readlines()]
    stock_code = temp_stock_code_list[0]
    del temp_stock_code_list
    
    
    #csvからDataFrameを読み込み
    input_original_df = pd.read_csv(csv_path, encoding='utf-8')
    temp_df = input_original_df.loc[:, ["Date", "Time", "Open","High", "Low", "Close"]]
    temp_df["Vol"] = input_original_df.loc[:, ["Up"]]#Upを一時的にVolに

    print (stock_code,"の処理中")

    available_times =  ["09:10", "09:20", "09:30", "09:40", "09:50", "10:00",
                        "10:10", "10:20", "10:30", "10:40", "10:50", "11:00",
                        "11:10", "11:20", "11:30",
                        "12:40", "12:50", "13:00",
                        "13:10", "13:20", "13:30", "13:40", "13:50", "14:00",
                        "14:10", "14:20", "14:30", "14:40", "14:50", "15:00"]
    
    #available_timesを含むデータのみを抽出
    temp_available_index_list = []
    for temp_time in available_times:
        available_index = temp_df.index[temp_df.Time == temp_time]
        for index in available_index:
            temp_available_index_list.append(index)
    temp_available_index_list.sort()
    temp_df = temp_df.loc[temp_available_index_list, :]
    del temp_available_index_list
    #temp_df.to_csv(folders["temp"]+"temp_scv_{}.csv".format(temp_cnt)) #加工後のcsv元データは必要か？


    #１日あたり３０-missing_values_range件のデータが揃っている物を抽出(揃ってないものを省く)
    c = collections.Counter(temp_df.loc[:, "Date"])
    unavailable_day_list = [k for k, v in c.items() if v < (len(available_times)-missing_values_range)] #欠損の猶予なしmissing_values_range = 0の時
    
    temp_delete_index_list = []
    for temp_day in unavailable_day_list:
        delete_index = temp_df.index[temp_df.Date == temp_day]
        for temp_index in delete_index:
            temp_delete_index_list.append(temp_index)
    temp_df.drop(temp_delete_index_list, inplace=True)
    
    
    
    #ここで１０刻みに穴があるものは埋める必要あり
    #欠損値を埋めて、収益率変換(30個そろった状態で出力)
    
    temp_x_df = calculate_df(temp_df, 1, management_flag=1)
    del temp_df
    temp_x_df.to_csv("./temp/{}_raw_data_{}.csv".format(stock_code, missing_values_range))
    print("{}_raw_data_{}.csv".format(stock_code, missing_values_range),"を出力しました。")
        
        
    exist_day_list = list(dict.fromkeys(np.array(temp_x_df.Date)))
    temp_x_data_2d = np.array(temp_x_df.loc[:, ['Open', 'High', 'Low', 'Close', 'Vol']])
    x_data_3d = temp_x_data_2d.reshape(-1,5,6,5)
    if(management_flag == 1): print("line600: ",len(exist_day_list) == len(x_data_3d))
    
    
    print("Yデータ用のファイルを準備してください。")
    temp_df = csv_to_df("",input_columns_info="Basic", output_y_info="All", 
                NK255_flag=0, x_interval_of_days=1, y_interval_of_days=1, zero_flag=0, management_flag=0)
    y_data = temp_df.loc[:, ["Date", "UpDown", "Benefit"]]
    
    len(y_data.Date)
        
    for temp_day in np.array(y_data.Date):
        exist_day_list.append(temp_day)
    

    c = collections.Counter(exist_day_list)
    unavailable_day_list = [k for k, v in c.items() if v < 2] #欠損の猶予なしmissing_values_range = 0の時
    
    temp_delete_index_list = []
    for temp_day in unavailable_day_list:
        delete_index = y_data.index[y_data.Date == temp_day]
        for temp_index in delete_index:
            temp_delete_index_list.append(temp_index)
    y_data.drop(temp_delete_index_list, inplace=True)
    if(management_flag == 1): print("line536: ",len(y_data) == len(x_data_3d))

    
    X_data = np.array(x_data_3d)
    Y_data = np.array(y_data.UpDown)
    
    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, train_size = 0.8, test_size = 0.2, shuffle = True)
    
    return x_train, y_train, x_test, y_test



def add_x_data(original_df, add_data_file_path, add_x_data_volume, x_interval_of_days, zero_flag):
    #データを読込
    temp_df = original_df
    temp_data_ori = pd.read_csv(add_data_file_path, encoding='utf-8')
    temp_data = temp_data_ori.loc[:, ["Open", "High", "Low", "Close", "Vol"]]
    #temp_data = temp_data_ori.loc[:, ["Date","Open", "High", "Low", "Close", "Vol"]]
    x_now = temp_data
    n=1
    while n <= add_x_data_volume:
        x_before = x_now.shift(int(x_interval_of_days+n))
        x_add_df = (x_now - x_before)/x_before
        #X_data["Date"] = temp_data_ori.Date
        x_add_df = x_add_df.shift(-x_interval_of_days)
        temp_df["Open{}".format(n)] = x_add_df.Open
        temp_df["High{}".format(n)] = x_add_df.High
        temp_df["Low{}".format(n)] = x_add_df.Low
        temp_df["Close{}".format(n)] = x_add_df.Close
        temp_df["Vol{}".format(n)] = x_add_df.Vol
        n += 1
    temp_df = temp_df.dropna(how='any') #念のため（なくても良いはず）
    return temp_df

#ファイルの型と拡張子を調べて、・booleanで出力
def type_and_extension_check(check_file):
    path_type = type(check_file)
    if(path_type == str):
        #拡張子のチェック
        file_extension_check = extension_check(check_file)
    elif(path_type == list):
        #リスト内のpathの拡張子をそれぞれ確認
        file_extension_check = list_extension_check(check_file)
    else:
        print("{}型が入力されています。".format(path_type))
        print("加工対象データpathは、str または list　型で入力してください。")
        file_extension_check = False
    return file_extension_check


#拡張子が、txtまたはcsvであることを確認してbooleanで出力
def extension_check(file_name):
    file_extension_check = ((".txt" in file_name) or (".csv" in file_name))
    if(file_extension_check == False):
        print("########################################")
        print("ERROR:")
        print("使用できる拡張子は、 .txt または .csvです。")
    else: pass
    return file_extension_check


#拡張子が、txtまたはcsvであることを確認してbooleanで出力
def list_extension_check(list_name):
    unavailable_path_list = []
    unavailable_path_cnt = 0
    for temp_path in list_name:
        file_extension_check = extension_check(temp_path)
        if(file_extension_check == False):
            unavailable_path_list.append(temp_path)
            unavailable_path_cnt += 1
        else: pass
    if(unavailable_path_cnt>0):
        print("以下の{}ファイルは使用できません。".format(unavailable_path_cnt))
        print(unavailable_path_list)
        file_extension_check = False
    else:
        file_extension_check = True
    return file_extension_check

    
if __name__ == "__main__":
    num = 0
    if(num == 0):
        print("################################################")
        print("Please use this file(txt to list) in other file")
        print("################################################")
    else:
        print("管理者用コード実行")
        #input_file_path = "../datasets/car/day/NK7203.txt"
        input_file_path = ""
        """
        input_file_path=["test.file.name"];input_columns_info="Basic"; output_y_info="All"
        NK255_flag=0; x_interval_of_days=1; y_interval_of_days=1; zero_flag=0; management_flag=1
        
        """
        #temp_data = csv_to_df(input_columns_info=0, zero_flag=0, output_y_info=2, NK255_flag=0, input_file_path=input_file_path, x_interval_of_days=1, y_interval_of_days=1, management_flag=1)
        #temp_data.to_csv("temp_stock_data.csv")
        #print(temp_data)
        csv_to_df(["test.file.name"],input_columns_info="Basic", output_y_info="All", 
                NK255_flag=0, x_interval_of_days=5, y_interval_of_days=5, zero_flag=0, management_flag=1)


### [EOF]