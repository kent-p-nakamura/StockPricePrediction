"""
spp_copy.py: バッチ処理にて必要なファイルをコピー
"""
import os
import shutil
import re


def main():  
    folders = {
                "temp": "./temp/",
                }
    
    files = {
            "spp": "../code/stock_price_prediction.py",
            "spp_cnn": "../code/stock_price_prediction_cnn.py",
            "modify": "../code/modify_the_data.py",
            "plot": "../code/plot_history.py",
            "misc": "../code/misc_box.py",
            "bat": "../code/spp_bat.py"
            }
    
    
    #初期フォルダの作成
    for name in folders:
        if not os.path.isdir(folders[name]):
            os.mkdir(folders[name])

    print("########################################")
    print("以下のFile(Folder)をcopyします。")
    for name in files:
        #一時ファイルとして保存
        file_name = re.findall("/code/\S*", files[name])[0].split("code/")[1]
        temp_file = "./"+ file_name
        shutil.copyfile(files[name], temp_file)
        print(files[name])
    
    print("copy完了")
    print("########################################")








if __name__ == '__main__':
    numM = 1
    if(numM == 0):
        print("################################################")
        print("Please use this file in other file")
        print("################################################")
    else:
        print("管理者用コード実行")
        main()






