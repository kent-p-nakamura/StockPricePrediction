"""
misc_box.py: 再利用モジュール
"""
import os


#初期フォルダの作成
def make_folders(folders):
    for name in folders:
        if not os.path.isdir(folders[name]):
            os.makedirs(folders[name])
            print("「{}」を作成しました。".format(folders[name]))
        else: pass

def make_folder(folder_path):
    if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
            print("「{}」を作成しました。".format(folder_path))
    else: pass

#使用中のファイル名を表示
def print_file_name(file_name):
    print("########################################")
    print("")
    print("Using current file(s), {}」".format(file_name))
    print("")
    print("########################################")

#ファイルpathを入力して、リスト化して出力
def input_file_list(out_string=None):
    temp_list = []
    if(out_string == None):
        out_string = "ファイル"
    else: pass
    print("########################################")
    print("")
    print("下に、{}をドラック＆ドロップして、エンターを押してください。".format(out_string))
    temp_path = input(">>> ")
    temp_path = temp_path.replace("'", "")
    temp_path = temp_path.replace(",", " ").split()
    for i in temp_path:
        temp_list.append(i)
    file_list = temp_list
    del temp_list
    return file_list

        
if __name__ == '__main__':
    print("################################################")
    print("this file is 'misc_box.py'")
    print("################################################")
