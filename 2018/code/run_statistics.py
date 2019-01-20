"""
run_statistics.py: 学習結果ログから分析
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import shutil
import re
from tqdm import tqdm, trange
from time import sleep
import cv2
from glob import glob



#関係するフォルダ
folders = {
            "statistics": "../statistics/",
            "temp": "./temp/",
            "raw": "../statistics/raw/",
            "basic": "../statistics/basic/"
            }

#初期フォルダの作成
for name in folders:
    if not os.path.isdir(folders[name]):
        os.makedirs(folders[name])


analysis_index = [
        "Statistics",
       "File Name", 
        "Samples",
       "corr.",
       "acc:",
       "mean",
        "50%",
        "mode",
        "std",
        "min",
        "max",
        "benefit:",
       "mean",
        "50%",
        "mode",
        "std",
        "min",
        "max"
        ]


def basic_statistics():
     #使用データ準備
    temp_list = []
    print("########################################")
    print("")
    print("Main")
    print('下に、任意のテキストファイルをドラック＆ドロップしてエンターを。')
    temp = input('>>> ')
    temp = temp.replace("'", "")
    temp = temp.replace(",", " ").split()
    for j in temp:
        temp_list.append(j)
    print("")
    print("########################################")
    print("")
    print("use this file(s)({})".format(str(temp_list)))
    print("")
    print("########################################")  
    sleep(0.5)
    
    
    
    analysis_index.reverse()
    for i in tqdm(temp_list):
        #一時ファイルとして保存
        #i = temp_list[0]
        file_name = re.findall("stock\S*", i)[0][:-4]
        temp_file = folders["temp"]+"temp_statistics.txt"
        shutil.copyfile(i, temp_file)

        if(temp_file.find(".txt")):
            before = temp_file
            after = temp_file.replace(".txt", ".csv")
            os.rename(before, after)
            temp_file = after
        df = pd.read_csv(temp_file, encoding="utf-8")
        
        sRow_df = df[df["aveFlag"] == 0] #all row data
        sRow_df = sRow_df.loc[:, sRow_df.columns != "aveFlag"]
        s_df = df[df["aveFlag"] == 1] # all average data
        s_df = s_df.loc[:, s_df.columns != "aveFlag"]
        
        make_figure(sRow_df, folders["raw"], file_name+"_raw", analysis_index)
        make_figure(s_df, folders["basic"], file_name, analysis_index)
        os.remove(temp_file)
    print("")
    print("終了しました。")
        
        
def make_figure(df, folder, name, analysis_index):
    #df = sRow_df
    #name = file_name+"_raw"
    #分析準備
    str1 = analysis_index
    str2 = []
    str2.append("")
    str2.append(name) #"File名"
    str2.append(len(df)) #"Sample数"
    df_corr = df.corr()
    str2.append(df_corr.loc["acc", "benefit"])# "相関係数",
    d=df.describe()
    str2.append("")
    str2.append(d.acc[1]) #平均
    str2.append(d.acc[5]) # 中央値
    mode1 = df.acc.mode()
    if(len(mode1)==1):
        str2.append(mode1[0]) # 最頻値
    else:
        str2.append("-")
    str2.append(d.acc[2])#標準偏差
    str2.append(d.acc[3]) #最小値
    str2.append(d.acc[7]) #最大値
    str2.append("")
    str2.append(d.benefit[1]) #平均
    str2.append(d.benefit[5]) # 中央値
    mode2 = df.benefit.mode()
    if(len(mode2)==1):
        str2.append(mode2[0]) # 最頻値
    else:
        str2.append("-")
    str2.append(d.benefit[2])#標準偏差
    str2.append(d.benefit[3]) #最小値
    str2.append(d.benefit[7]) #最大値
    fsz = 12
    
    
    
    
    plt.style.use("default")
    sns.set()
    sns.set_style("whitegrid")
    acc = df.acc
    benefit = df.benefit
    #bins = int(math.log(len(df), 2)+1)    
    
    fig, (axL, axM, axR, axS) = plt.subplots(ncols=4, figsize=(28,6))
    
    
    # plot
    axL.scatter(acc, benefit, alpha=0.5, linewidths="2")
    # x axis
    axL.set_xlabel("acc [%]")
    # y axis
    axL.set_ylabel("benefit [yen/day]")
    # title
    axL.set_title("scatter Acc-Benefit")
    
    #axM
    acc_bins = np.arange(46, 60, 0.5)
    axM.hist(acc, bins=acc_bins, rwidth=0.9)
    #axM.hist(acc, bins=28, rwidth=0.9)
    # x axis
    axM.set_xlabel("class")    
    # y axis
    #plt.ylim([0,len(df)/6])
    axM.set_ylabel("frequency")   
    # title
    axM.set_title("Accuracy [%]")
    
    #axR
    benefit_bins = np.arange(-7, 7, 0.5)
    axR.hist(benefit, bins=benefit_bins, rwidth=0.9)
    #axR.hist(benefit, bins=28, rwidth=0.9)
    # x axis
    axR.set_xlabel("class")
    # y axis
    #plt.ylim([0,len(df)/6])
    axR.set_ylabel("frequency")
    # title
    axR.set_title("Benefit [yen/day]")
    
    #統計情報
    xmin=0; ymin=0
    xmax=5
    ymax=len(str1)
    #ymax=len(df)/6
    #str1.reverse()
    str2.reverse()
    
    plt.axis('off')
    #plt.xlim(xmin,xmax)
    #plt.ylim(ymin,ymax)
    for i in range(len(str2)):
        if(i == 6 or i == 13 or i == 14 or i == 15):
            axS.text(0.5,i+0.2,str1[i],ha='left',va='bottom',fontsize=fsz)
            axS.text(2.5,i+0.2,str2[i],ha='left',va='bottom',fontsize=fsz)
        elif len(str2)-2<=i:
            axS.text(1.0,i+0.2,str1[i],ha='center',va='bottom',fontsize=fsz)
            axS.text(3.5,i+0.2,str2[i],ha='center',va='bottom',fontsize=fsz)
        else:
            axS.text(1.5,i+0.2,str1[i],ha='right',va='bottom',fontsize=fsz)
            axS.text(2.5,i+0.2,str2[i],ha='left',va='bottom',fontsize=fsz)
    seqy=np.array([ymax,ymax-2,ymin])
    axS.hlines(seqy, xmin, xmax)
    axS.vlines(2,ymin, ymax, linestyle='dashed')
    
    
    # save as png
    plt.savefig(folder+name)
    #plt.show()
    plt.close()


def compared_figure():
    temp_list = []
    print("########################################")
    print("")
    print("Compare")
    print('下に、比較するテキストファイルをドラック＆ドロップしてエンターを。')
    temp = input('>>> ')
    temp = temp.replace("'", "")
    temp = temp.replace(",", " ").split()
    for j in temp:
        temp_list.append(j)
    print("")
    print("########################################")
    print("")
    print("use this file(s)({})".format(str(temp_list)))
    print("")
    print("########################################")  
    sleep(0.5)
    

    
    
    analysis_index.reverse()
    i = temp_list[0]
    #一時ファイルとして保存
    #i = temp_list[0]
    file_name = re.findall("stock\S*", i)[0][:-4]
    temp_file = folders["temp"]+"temp_statistics.txt"
    shutil.copyfile(i, temp_file)

    if(temp_file.find(".txt")):
        before = temp_file
        after = temp_file.replace(".txt", ".csv")
        os.rename(before, after)
        temp_file = after
    df = pd.read_csv(temp_file, encoding="utf-8")
    
    sRow_df = df[df["aveFlag"] == 0] #all row data
    sRow_df = sRow_df.loc[:, sRow_df.columns != "aveFlag"]
    s_df = df[df["aveFlag"] == 1] # all average data
    s_df = s_df.loc[:, s_df.columns != "aveFlag"]
    
    
    
    
    i = temp_list[1]
    #一時ファイルとして保存
    #i = temp_list[0]
    file_name2 = re.findall("stock\S*", i)[0][:-4]
    temp_file = folders["temp"]+"temp_statistics.txt"
    shutil.copyfile(i, temp_file)

    if(temp_file.find(".txt")):
        before = temp_file
        after = temp_file.replace(".txt", ".csv")
        os.rename(before, after)
        temp_file = after
    df = pd.read_csv(temp_file, encoding="utf-8")
    
    sRow_df2 = df[df["aveFlag"] == 0] #all row data
    sRow_df2 = sRow_df2.loc[:, sRow_df2.columns != "aveFlag"]
    s_df2 = df[df["aveFlag"] == 1] # all average data
    s_df2 = s_df2.loc[:, s_df2.columns != "aveFlag"]
    
    
    
    make_figure2(sRow_df, sRow_df2, folders["raw"], file_name+"_raw",file_name2+"_raw", analysis_index)
    make_figure2(s_df, s_df2, folders["basic"], file_name,file_name2, analysis_index)
    os.remove(temp_file)
    print("")
    print("終了しました。")

def make_figure2(df, df2, folder, name,name2, analysis_index):
    #df = sRow_df
    #name = file_name+"_raw"
    #分析準備
    str1 = analysis_index
    str2 = []
    str2.append("")
    str2.append(name+"(Blue)") #"File名"
    str2.append(len(df)) #"Sample数"
    df_corr = df.corr()
    str2.append(df_corr.loc["acc", "benefit"])# "相関係数",
    d=df.describe()
    str2.append("")
    str2.append(d.acc[1]) #平均
    str2.append(d.acc[5]) # 中央値
    mode1 = df.acc.mode()
    if(len(mode1)==1):
        str2.append(mode1[0]) # 最頻値
    else:
        str2.append("-")
    str2.append(d.acc[2])#標準偏差
    str2.append(d.acc[3]) #最小値
    str2.append(d.acc[7]) #最大値
    str2.append("")
    str2.append(d.benefit[1]) #平均
    str2.append(d.benefit[5]) # 中央値
    mode2 = df.benefit.mode()
    if(len(mode2)==1):
        str2.append(mode2[0]) # 最頻値
    else:
        str2.append("-")
    str2.append(d.benefit[2])#標準偏差
    str2.append(d.benefit[3]) #最小値
    str2.append(d.benefit[7]) #最大値
    
    
    str3 = []
    str3.append("")
    str3.append(name2+"(Red)") #"File名"
    str3.append(len(df2)) #"Sample数"
    df_corr = df2.corr()
    str3.append(df_corr.loc["acc", "benefit"])# "相関係数",
    d=df2.describe()
    str3.append("")
    str3.append(d.acc[1]) #平均
    str3.append(d.acc[5]) # 中央値
    mode1 = df2.acc.mode()
    if(len(mode1)==1):
        str3.append(mode1[0]) # 最頻値
    else:
        str3.append("-")
    str3.append(d.acc[2])#標準偏差
    str3.append(d.acc[3]) #最小値
    str3.append(d.acc[7]) #最大値
    str3.append("")
    str3.append(d.benefit[1]) #平均
    str3.append(d.benefit[5]) # 中央値
    mode2 = df2.benefit.mode()
    if(len(mode2)==1):
        str3.append(mode2[0]) # 最頻値
    else:
        str3.append("-")
    str3.append(d.benefit[2])#標準偏差
    str3.append(d.benefit[3]) #最小値
    str3.append(d.benefit[7]) #最大値
    
    fsz = 12
    
    
    
    
    plt.style.use("default")
    sns.set()
    sns.set_style("whitegrid")
    acc = df.acc
    benefit = df.benefit
    acc2 = df2.acc
    benefit2 = df2.benefit
    #bins = int(math.log(len(df), 2)+1)    
    
    fig, (axL, axM, axR, axS, axS2) = plt.subplots(ncols=5, figsize=(32,6))
    
    
    # plot
    axL.scatter(acc, benefit, color="blue", alpha=0.5, linewidths="2")
    axL.scatter(acc2, benefit2, color="red", alpha=0.5, linewidths="2")
    # x axis
    axL.set_xlabel("acc [%]")
    # y axis
    axL.set_ylabel("benefit [yen/day]")
    # title
    axL.set_title("scatter Acc-Benefit")
    
    #axM
    acc_bins = np.arange(46, 60, 0.5)
    axM.hist(acc, color="blue", alpha=0.5, bins=acc_bins, rwidth=0.9)
    axM.hist(acc2, color="red", alpha=0.5, bins=acc_bins, rwidth=0.9)
    # x axis
    axM.set_xlabel("class")    
    # y axis
    #plt.ylim([0,len(df)/6])
    axM.set_ylabel("frequency")   
    # title
    axM.set_title("Accuracy [%]")
    
    #axR
    benefit_bins = np.arange(-7, 7, 0.5)
    axR.hist(benefit, color="blue", alpha=0.5, bins=benefit_bins, rwidth=0.9)
    #axR.hist(benefit, color="blue", alpha=0.5, bins=14, rwidth=0.9)
    axR.hist(benefit2, color="red", alpha=0.5, bins=benefit_bins, rwidth=0.9)
    #axR.hist(benefit2, color="red", alpha=0.5, bins=14, rwidth=0.9)
    # x axis
    axR.set_xlabel("class")
    # y axis
    #plt.ylim([0,len(df)/6])
    axR.set_ylabel("frequency")
    # title
    axR.set_title("Benefit [yen/day]")
    
    #統計情報
    xmin=0; ymin=0
    xmax=5
    ymax=len(str1)
    #ymax=len(df)/6
    #str1.reverse()
    str2.reverse()
    
    axS.axis("off")
    #plt.xlim(xmin,xmax)
    #plt.ylim(ymin,ymax)
    for i in range(len(str2)):
        if(i == 6 or i == 13 or i == 14 or i == 15):
            axS.text(0.5,i+0.2,str1[i],ha='left',va='bottom',fontsize=fsz)
            axS.text(2.5,i+0.2,str2[i],ha='left',va='bottom',fontsize=fsz)
        elif len(str2)-2<=i:
            axS.text(1.0,i+0.2,str1[i],ha='center',va='bottom',fontsize=fsz)
            axS.text(3.5,i+0.2,str2[i],ha='center',va='bottom',fontsize=fsz)
        else:
            axS.text(1.5,i+0.2,str1[i],ha='right',va='bottom',fontsize=fsz)
            axS.text(2.5,i+0.2,str2[i],ha='left',va='bottom',fontsize=fsz)
    seqy=np.array([ymax,ymax-2,ymin])
    axS.hlines(seqy, xmin, xmax)
    axS.vlines(2,ymin, ymax, linestyle='dashed')
    
    
    #統計情報
    #xmin=0; ymin=0
    #xmax=5
    #ymax=len(str1)
    #ymax=len(df)/6
    #str1.reverse()
    str3.reverse()
    
    axS2.axis('off')
    #plt.xlim(xmin,xmax)
    #plt.ylim(ymin,ymax)
    for i in range(len(str3)):
        if(i == 6 or i == 13 or i == 14 or i == 15):
            axS2.text(0.5,i+0.2,str1[i],ha='left',va='bottom',fontsize=fsz)
            axS2.text(2.5,i+0.2,str3[i],ha='left',va='bottom',fontsize=fsz)
        elif len(str3)-2<=i:
            axS2.text(1.0,i+0.2,str1[i],ha='center',va='bottom',fontsize=fsz)
            axS2.text(3.5,i+0.2,str3[i],ha='center',va='bottom',fontsize=fsz)
        else:
            axS2.text(1.5,i+0.2,str1[i],ha='right',va='bottom',fontsize=fsz)
            axS2.text(2.5,i+0.2,str3[i],ha='left',va='bottom',fontsize=fsz)
    seqy=np.array([ymax,ymax-2,ymin])
    axS2.hlines(seqy, xmin, xmax)
    axS2.vlines(2,ymin, ymax, linestyle='dashed')
    
    # save as png
    plt.savefig(folder+name)
    #plt.show()
    plt.close()



def make_2d_image():
    
     #使用データ準備
    temp_list = []
    print("########################################")
    print("")
    print('下に、任意のテキストファイルをドラック＆ドロップしてエンターを。')
    temp = input('>>> ')
    temp = temp.replace("'", "")
    temp = temp.replace(",", " ").split()
    for j in temp:
        temp_list.append(j)
    print("")
    print("########################################")
    print("")
    print("use this folder(s)({})".format(str(temp_list)))
    print("")
    print("########################################")  
    sleep(0.5)
          
    #関係するフォルダ
    folders = {
                "history": "../historyImg/",
                "temp": "./temp/",
                "combine": "../historyImg/combine/"
                }

    #初期フォルダの作成
    for name in folders:
        if not os.path.isdir(folders[name]):
            os.mkdir(folders[name])
       
    #出力
    for h in temp_list:
        png_files = glob(h+"history*")
        png_files.sort()
        temp_loc = png_files[0][:-10]
        im_list_2d = []
        s = int(png_files[0][-9:-8])
        g = s + int(len(png_files)/25)  #sからの増加分をたす。
        for i in trange(s, g, 1, desc="loop"):
            im_list_v = []
            for j in range(1,6,1):
                im_list_h = []
                for k in range(1,6,1):
                    path = temp_loc+"_{}_{}_{}.png".format(i,j,k)
                    #print(path)
                    im_list_h.append(cv2.imread(path))
                im_list_v.append(im_list_h)              
            im_list_2d = im_list_v
            
            im_2d = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])
            
            out_folder = temp_loc.split("/history_")[0]+"/"
            out_folder = folders["combine"]+out_folder.split("historyImg/")[1]
            if not os.path.isdir(out_folder):
                os.makedirs(out_folder)
            out_fName = out_folder+re.findall("history_\S*", temp_loc)[0]+"_{}.png".format(i)
            cv2.imwrite(out_fName, im_2d)
        
        #使用したファイルを削除
        shutil.rmtree(h)



if __name__ == '__main__':
    numM = 1
    if(numM == 0):
        print("################################################")
        print("Please use this file in other file")
        print("################################################")
    else:
        print("########################################")
        print("")
        print('compareなら「１」を入力。それ以外は単純分析。')
        aaa=input('>>> ')
        if(aaa == "1"):
            compared_figure()
        elif(aaa == "2"):
            make_2d_image()
        else:
            basic_statistics()





### [EOF]