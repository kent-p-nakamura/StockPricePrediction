"""
spp_del.py: バッチ処理後、ファイル消去
"""
import os
import shutil
from glob import glob

def main():
    exist_files = glob("./*")
    
    print("########################################")
    print("以下のFile(Folder)を削除します。")
    for file in exist_files:
        if not (("spp_copy.py" in file) or ("spp_del.py" in file) or ("run_spp_main.bat" in file)):
            if(os.path.isfile(file)):
                os.remove(file)
                print(file)
            elif(os.path.isdir(file)):
                shutil.rmtree(file)
                print(file)
            else:
                print("ERROR: {}は、file, dirではありません。".format(file))            
    print("削除完了")
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
