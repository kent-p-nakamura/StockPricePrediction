"""
stock_price_prediction.py: 株価予想、実験モデル１
"""
from keras.utils import np_utils
#from tensorflow.python.keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import model_from_json
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import os
import os.path
from statistics import mean
#
import modify_the_data as mtd
import plot_history as ph
import misc_box

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" #Default is 0, abailable from 0 to 3

#関係するフォルダ
folders = {
            "model": "../modelWeight/",
            "log": "../logs/",
            "history": "../historyImg/",
            "datasets": "../datasets/",
            "temp": "./temp/"
            }

#初期フォルダの作成
misc_box.make_folders(folders)


initializers = {
        0 : "normal",
        1 : "uniform",
        2 : "lecun_uniform",
        3 : "identity",
        4 : "orthogonal",
        5 : "zero",
        6 : "glorot_normal",
        7 : "glorot_uniform",
        8 : "he_normal",
        9 : "he_uniform"
        }

activations = {
        0 : "tanh", #ハイパボリックタンジェント
        1 : "softsign", #ソフトサイン
        2 : "sigmoid", #シグモイド関数
        3 : "hard_sigmoid", #ハードシグモイド
        4 : "softplus", #ソフトプラス
        5 : "relu", #ランプ関数
        6 : "linear",
        7 : "softmax",
        8 : "elu",
        9 : "selu"
        }

optimizers = {
        0 : "adam",
        1 : "sgd",
        2 : "rmsprop",
        3 : "adagrad",
        4 : "adadelta",
        5 : "adamax",
        6 : "nadam"
        }

objectives = {
        0 : "binary_crossentropy", #2クラス分類時の交差エントロピー
        1 : "categorical_crossentropy", #Nクラス分類時の交差エントロピー
        2 : "mse", #平均二乗誤差
        3 : "mae", #平均絶対誤差
        #4 : "mspa", #平均絶対誤差率
        5 : "msle", #対数平均二乗誤差
        6 : "hinge", #ヒンジ損失の和
        7 : "squared_hinge", #ヒンジ損失の二乗の和
        8 : "sparse_categorical_crossentropy", #スパースなNクラス分類交差エントロピー
        9 : "kld", #KLダイバージェンス
        10 : "cosine_proximity", #コサイン類似度を負にしたもの
        11 : "kullback_leibler_divergence",
        12 : "poisson"
        }



def main(learning_cnt=0, learning_data_file_path="", industry_flag=0, zero_flag=0, add_x_data_volume=0, drop_train_percent=0, 
                                     x_interval_of_days=1, y_interval_of_days=1, prediction_y_info="UpDown", management_flag=1):
    """
    #learning_cnt    mainの施行回数
    #prediction_y_info 予測したい値 0 ⇐ Weather
    #                  1 ⇐ UpDown
    #industry_flag 会社数     0 ⇐１社の場合
    #                 1 ⇐　多数の会社の場合
    #zero_flag            0: 
    #                 1: ０を区別
    """
    #各種パラメータ
    n_splits=5 #KFoldの分割数
    
    units = 20
    batch_size = 20
    epochs = 200
    verbose=0
    
    #Dense
    initializer = initializers[0] # 7, *8
    activationDen = activations[0] # *0, 5, 6 9
    activationOut = activations[7]
    #compile
    optimizer = optimizers[0] # *0 3
    loss = objectives[0] #0 5 *12 not8
    if(prediction_y_info=="Weather" or zero_flag!=0): loss = objectives[1]
    ##EarlyStoppingについて
    patienceE=10 #最低ループ数
    verboseE=0
    ##ReduceLROnPlateauについて
    factor=0.2 #学習率を減らす割合
    patienceR=10 #学習率削減の頻度（epoch基準）
    cooldown=5 #学習率低減後、通常学習率に戻るまでのepoch数
    min_lr=0.001 #学習率の下限
    verboseR=0
    
    verbose = verboseE = verboseR = 1 #強制OnOff
    
    
        
    #model_folder_path, history_folder_path準備
    model_folder_path, history_folder_path = make_folder_path(prediction_y_info, drop_train_percent, add_x_data_volume)
    
    #各種セーブファイル名の指定
    log_folder = folders["log"]+"{}_{}/".format(x_interval_of_days, y_interval_of_days)
    misc_box.make_folder(log_folder)
    if(industry_flag == 0): #一社だけで学習
        if(prediction_y_info == "Weather"):
            model_file_name = model_folder_path + "stockWeather_{}".format(learning_cnt)
            model_name = "stockWeather_{}_model".format(learning_cnt)
            log_file_name = log_folder+"stockWeather_{}_{}".format(x_interval_of_days, y_interval_of_days)
            history_title = "Weather"
        elif(prediction_y_info == "UpDown"):
            model_file_name = model_folder_path+"stockUpDown_{}".format(learning_cnt)
            model_name = "stockUpDown_{}_model".format(learning_cnt)
            log_file_name = log_folder+"stockUpDown_{}_{}".format(x_interval_of_days, y_interval_of_days)
            history_title = "UpDown"
        else: pass
    elif(industry_flag == 1): #複数社のデータで学習
        if(prediction_y_info == "Weather"):
            model_file_name = model_folder_path+"stockWeather_com_{}".format(learning_cnt)
            model_name = "stockWeather_com_{}_model".format(learning_cnt)
            log_file_name = log_folder+"stockWeather_com_{}_{}".format(x_interval_of_days, y_interval_of_days)
            history_title = "Weather_com"
        elif(prediction_y_info == "UpDown"):
            model_file_name = model_folder_path+"stockUpDown_com_{}".format(learning_cnt)
            model_name = "stockUpDown_com_{}_model".format(learning_cnt)
            log_file_name = log_folder+"stockUpDown_com_{}_{}".format(x_interval_of_days, y_interval_of_days)
            history_title = "UpDown_com"
        else: pass
    
    #log_file_name準備
    if(add_x_data_volume > 0): log_file_name = log_file_name+"_add{}".format(add_x_data_volume)
    if(drop_train_percent > 0): log_file_name = log_file_name+"_drop{}".format(drop_train_percent)
    log_file_name = log_file_name+".txt"
    
    
    
    #data準備
    if(learning_cnt == 0):
        #学習データの作成・保存・ロード
        temp_stock_data = mtd.csv_to_df(learning_data_file_path, input_columns_info="Basic", output_y_info="All", 
                NK255_flag=0, x_interval_of_days=x_interval_of_days, y_interval_of_days=y_interval_of_days, zero_flag=zero_flag, management_flag=0)
        temp_stock_data.to_csv(folders["temp"]+"temp_stock_data.csv")
        del temp_stock_data
        stock_original_df = pd.read_csv(folders["temp"]+"temp_stock_data.csv", encoding='utf-8')
        stock_original_df = stock_original_df.loc[:, stock_original_df.columns != "Unnamed: 0"]        
    else:
        #学習データのロード
        stock_original_df = pd.read_csv(folders["temp"]+"temp_stock_data.csv", encoding='utf-8')
        stock_original_df = stock_original_df.loc[:, stock_original_df.columns != "Unnamed: 0"]
    
    #xデータを追加と保存
    if(add_x_data_volume > 0 and learning_cnt == 0):
        with open(folders["temp"]+"temp_csv_path.txt", 'r') as f:
            temp_csv_path_list = [s.strip() for s in f.readlines()]
        learning_data_file_path = temp_csv_path_list[0]
        del temp_csv_path_list 
        stock_original_df = mtd.add_x_data(stock_original_df, learning_data_file_path, add_x_data_volume, x_interval_of_days, zero_flag)
        stock_original_df.to_csv(folders["temp"]+"temp_stock_data.csv")
    else: pass   

    #xデータ準備
    temp_df = stock_original_df
    temp_df = temp_df.loc[:, temp_df.columns != "Date"]
    temp_df = temp_df.loc[:, temp_df.columns != "UpDown"]
    temp_df = temp_df.loc[:, temp_df.columns != "Weather"]
    temp_df = temp_df.loc[:, temp_df.columns != "rClose"]
    x_data = temp_df.loc[:, temp_df.columns != "Benefit"]
    del temp_df
    if(management_flag == 1): print(x_data.columns)
    #yデータ準備
    if(prediction_y_info == "Weather"): #Weather
        y_data = stock_original_df.loc[:, ["Weather", "Benefit", "rClose"]]
    elif(prediction_y_info == "UpDown"): #UpDown
        y_data = stock_original_df.loc[:, ["UpDown", "Benefit", "rClose"]]
    else: pass
    if(management_flag == 1): print(y_data.columns)
    

    nb_classes = y_data.iloc[:,0].nunique() #Yデータの種類数
    X = np.array(x_data)
    Y = np.array(y_data)
    
    
    
    testScores = [] #testデータによる検証結果について、acc, lossを保存
    benefit_log_list = [] #testデータによる検証結果について、収益に関する各種データを保存

    kf_out = KFold(n_splits, shuffle=True)
    kf_in = KFold(n_splits, shuffle=True)
    
    index_out = 0
    for train_group_index, test_index in kf_out.split(X):
        index_out = index_out+1
        x_test = X[test_index]
        y_test = np_utils.to_categorical(Y[test_index][:, 0], nb_classes).astype(np.int64)
        
        X_train_group = X[train_group_index]
        Y_train_group = Y[train_group_index]
        
        index = 0
        for train_index, eval_index in kf_in.split(X_train_group):
            index = index+1
            if(management_flag==1): print("{}回目: {}-{}".format(learning_cnt, index_out, index))
            if(drop_train_percent > 0):
                x_train, y_train = mtd.drop_train_group(X_train_group[train_index], Y_train_group[train_index], drop_train_percent)
                y_train = np_utils.to_categorical(y_train[:, 0], nb_classes).astype(np.int64)
                x_eval = X_train_group[eval_index]
            else:
                x_train, x_eval = X_train_group[train_index], X_train_group[eval_index]
                y_train = np_utils.to_categorical(Y_train_group[train_index][:, 0], nb_classes).astype(np.int64)
            y_eval = np_utils.to_categorical(Y_train_group[eval_index][:, 0], nb_classes).astype(np.int64)
        
            model_weights = model_file_name+"_weights_%d_%d.h5" % (index_out, index)
    
            in_size = len(x_train[0])
            out_size = nb_classes
            
            #Dense(units, activation='relu').get_weights()
            
            model = Sequential()
            model.add(Dense(units, kernel_initializer=initializer, activation=activationDen, input_shape=(in_size,))) #init='normal'
            model.add(Dropout(0.2))
            model.add(Dense(units, kernel_initializer=initializer, activation=activationDen))
            model.add(Dropout(0.2))
            model.add(Dense(units, kernel_initializer=initializer, activation=activationDen))
            model.add(Dropout(0.2))
            model.add(Dense(out_size, activation=activationOut)) #'softmax'
            
            model.compile(
                    loss = loss,
                    optimizer = optimizer,
                    metrics = ['accuracy']
                    )
            #学習率の再設定
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patienceR,
                                          cooldown=cooldown, min_lr=min_lr, verbose=verboseR)
            early_stopping = EarlyStopping(monitor='val_loss', patience=patienceE , verbose=verboseE)
            #checkpointer = ModelCheckpoint(model_weights, monitor='val_loss', verbose=0, save_best_only=True)
            hist = model.fit(x_train, y_train,
                             batch_size,
                             epochs,
                             verbose=verbose,
                             validation_data=(x_eval, y_eval),
                             #validation_split=0.2, #入力データの最後からの指定割合をvalidationとして使用
                             shuffle=True,
                             callbacks=[early_stopping
                                        #,checkpointer
                                        ,reduce_lr
                                        ])
            #model の保存
            '''
            json_string = model.to_json()
            json_name = model_name + ".json"
            open(os.path.join(model_folder_path, json_name),"w").write(json_string)
            '''
            #modelの画像保存
            '''
            model.summary()
            if(learning_cnt==0 and index==1): plot_model(model, to_file = model_folder_path+model_name+".png", 
                                                    show_layer_names=False, show_shapes=True)
            '''
            #historyチャート(acc, loss)表示
            '''
            inN = "{}_{}".format(index_out, index)
            ph.plot_history(hist, history_folder_path, learning_cnt, inN, history_title)
            '''
            #正答率・損失　表示
            score = model.evaluate(x_test, y_test, verbose = verbose)
            score[0] = round(score[0], 4)
            score[1] = round(score[1]*100, 4)
            testScores.append([score[1], score[0]])
            if(management_flag==1): print('正解率= ', score[1], 'loss= ', score[0])
    
            #マトリクス表示
            y_pred = model.predict_classes(x_test, verbose=0)
            y_test2 = []
            for i in range(y_test.shape[0]):
                y_test2.append(np.argmax(y_test[i]))
            cm = pd.DataFrame(confusion_matrix(y_test2, y_pred))
            if(prediction_y_info=="UpDown" and zero_flag==0):
                cm.index = ["Down", "Up"]
                cm.columns = ["pred.Down", "pred.Up"]
            if(management_flag==1):
                print(cm)
                print("")
                print(classification_report(y_test2, y_pred))
            else: pass
            
            
            #その他分析
            i = 0
            benefit = 0
            benefit_row = []
            UpDown_acc=0
            if(prediction_y_info=="UpDown"):
                while i < len(y_pred):
                    if (y_pred[i] == y_test2[i]):
                        benefit += abs(Y[test_index][i, 1])
                        benefit_row.append(abs(Y[test_index][i, 1]))
                    else:
                        benefit -= abs(Y[test_index][i, 1])
                        benefit_row.append(-abs(Y[test_index][i, 1]))
                    i += 1
            elif(prediction_y_info=="Weather"):
                temp_cnt=0
                temp_fit=0 #正解した数
                while i < len(y_pred):
                    temp_cnt+=1
                    #教師データがdownよりのとき
                    if(y_test2[i]==0 or y_test2[i]==1 or y_test2[i]==4):
                        if (y_pred[i]==0 or y_pred[i]==1 or y_pred[i]==4):
                            benefit += abs(Y[test_index][i, 1])
                            benefit_row.append(abs(Y[test_index][i, 1]))
                            temp_fit+=1
                        else:
                            benefit -= abs(Y[test_index][i, 1])
                            benefit_row.append(-abs(Y[test_index][i, 1]))
                    #教師データがupよりのとき
                    else:   
                        if (y_pred[i]==2 or y_pred[i]==3):
                            benefit += abs(Y[test_index][i, 1])
                            benefit_row.append(abs(Y[test_index][i, 1]))
                            temp_fit+=1
                        else:
                            benefit -= abs(Y[test_index][i, 1])
                            benefit_row.append(-abs(Y[test_index][i, 1]))
                    i += 1
                UpDown_acc=round((temp_fit/temp_cnt)*100, 5)
                
            else: print("ERROR: line364")
            test_days = len(y_pred)
            ave_stock = round(mean(Y[test_index][:, 2]), 1)
            total_benefit = round(benefit, 4)
            benefit = round(benefit/len(y_pred), 4)
            if(management_flag==1):
                print("UpDown換算にて、正答率= ", UpDown_acc)
                print("########################################")
                print("{}回目: {}-{} [RESULT]".format(learning_cnt, index_out, index))
                print("ランダムの", test_days,"日分のデータを使用し、")
                print("平均株価", ave_stock, "円のとき、")
                print("総収益は、",total_benefit,"円で、")
                print("1株売買で1日あたり収益は、",benefit,"円です。")
            else: pass
            day = -1
            year = -1
            if(benefit>0.0000):
                day = ave_stock / benefit
                day = round(day, 1)
                year = round(day/245, 1)
                if(management_flag==1): print(day,"日[",year,"年]で投資額を回収予定。（１年を２４５日営業とする。）")
            else: pass
            benefit_log_list.append([test_days, ave_stock, total_benefit, benefit, day, year, UpDown_acc])
            if(management_flag==1): print("########################################")
                  
    testScores = np.array(testScores)
    benefit_log_list = np.array(benefit_log_list)
    #print(testScores)
    acc = round(np.mean(testScores[:,0]),4)
    acc_std = round(np.std(testScores[:,0]),4)
    acc2 = round(np.mean(benefit_log_list[:,6]),4)
    loss = round(np.mean(testScores[:,1]),4)
    loss_std = round(np.std(testScores[:,1]),4)
    test_days = np.mean(benefit_log_list[:,0])
    ave_stock = round(np.mean(benefit_log_list[:,1]),1)
    total_benefit = round(np.mean(benefit_log_list[:,2]),4)
    benefit = round(np.mean(benefit_log_list[:,3]),4)
    day = -1
    year = -1
    if(management_flag==1):
        print("########################################")
        print("")
        print("{}回目:".format(learning_cnt))
        print("正答率: %.4f%% (+/- %.4f%%)" % (acc, acc_std))
        print("損失: %.4f (+/- %.4f)" % (loss, loss_std))
        print("単位収益: %.4f円 (+/- %.4f円)" % (benefit, np.std(benefit_log_list[:,3])))
        print("")
        print("########################################")
    else: pass

    
    if os.path.exists(log_file_name):
        f = open(log_file_name, 'a')
        f.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20}\n".format
                (learning_cnt,x_interval_of_days,y_interval_of_days,add_x_data_volume,drop_train_percent,n_splits,units,batch_size,epochs,test_days,acc,acc_std,acc2,
                 loss,loss_std,ave_stock,total_benefit,benefit,day,year,1))
    else:
        f = open(log_file_name, 'w')
        f.write(",x_interval_of_days,y_interval_of_days,add_x_data_volume,drop_train_percent,cv,units,batch_size,epochs,test_days,acc,acc_std,updown_acc,"+
                "loss,loss_std,ave_stock,total_benefit,benefit,day,year,aveFlag\n")
        f.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20}\n".format
                (learning_cnt,x_interval_of_days,y_interval_of_days,add_x_data_volume,drop_train_percent,n_splits,units,batch_size,epochs,test_days,acc,acc_std,acc2,
                 loss,loss_std,ave_stock,total_benefit,benefit,day,year,1))
    for i in range(0,n_splits**2):
        f.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20}\n".format
            (learning_cnt,"","","","","","","","","",testScores[i,0],"",benefit_log_list[i,6],
             testScores[i,1],"",benefit_log_list[i,1],benefit_log_list[i,2],benefit_log_list[i,3],"","",0))
    f.close()


#以下、要改良
def predict():
    #各種パラメータ
    #n_splits=5 #KFoldの分割数、学習した物によって毎回修正必要 
    test_pred = []
    test_average = 0
    X_test = x_test
    # 保存したモデル重みデータとモデルのロード
    print("#####################")
    for index in range(0, n_splits):
        model_weights = model_file_name+"_weights_%d.h5" % (index+1)
        json_name = model_name + ".json"
        model = model_from_json(open(os.path.join(model_folder_path, json_name)).read())
        model.load_weights(model_weights)
    
        # 各モデルにおける推測確率の計算　
        test_pred.append(model.predict(X_test))
        test_average += np.array(test_pred[index])
        if(index == 4): test_average /= 5


    #print(test_average)
    i = 0
    while i < len(test_average):
        if(test_average[i][0]>=0.5): print(test_average[i][0])
        i += 1
    
    
#model_folder_path, history_folder_path準備
def make_folder_path(prediction_y_info, drop_train_percent, add_x_data_volume):
    if(prediction_y_info == "Weather"): #Weather
        model_folder_path = folders["model"] +"Weather/"
        history_folder_path = folders["history"] +"Weather/"
    if(prediction_y_info == "UpDown"): #UpDown
        model_folder_path = folders["model"] +"UpDown/"
        history_folder_path = folders["history"] +"UpDown/"
    if((drop_train_percent > 0) and (add_x_data_volume > 0)):
        model_folder_path = model_folder_path + "add_drop/"
        history_folder_path = history_folder_path +"add_drop/{}_{}/".format(add_x_data_volume, drop_train_percent)
    else:
        if(add_x_data_volume > 0): 
            model_folder_path = model_folder_path +"add/{}/".format(add_x_data_volume)
            history_folder_path = history_folder_path +"add/{}/".format(add_x_data_volume)
        elif(drop_train_percent > 1): 
            model_folder_path = model_folder_path +"drop/{}/".format(drop_train_percent)
            history_folder_path = history_folder_path +"drop/{}/".format(drop_train_percent)
        else:
            model_folder_path = model_folder_path +"base/".format(drop_train_percent)
            history_folder_path = history_folder_path +"base/".format(drop_train_percent)
    if not os.path.isdir(model_folder_path):
        os.makedirs(model_folder_path)
    if not os.path.isdir(history_folder_path):
        os.makedirs(history_folder_path)
    return model_folder_path, history_folder_path



if __name__ == '__main__':
    numM = 0
    in_file=""
    if(numM == 0):
        print("################################################")
        print("Please use this file(stock learning) in other file")
        print("################################################")
    else:
        print("管理者用コード実行")
        """
        learning_cnt=0; industry_flag=0; zero_flag=0; add_x_data_volume=0; drop_train_percent=0
        x_interval_of_days=1; y_interval_of_days=1; prediction_y_info="UpDown"; management_flag=1 
        learning_data_file_path = "../datasets/car/day/NK7203.txt"
        learning_data_file_path = ""
        add_x_data_volume=4
        """
        #一例として、トヨタのデータを想定
        in_file = "../datasets/car/day/NK7203.txt" #NK****.txt, ****を証券コードとして記録することを推奨。
        main(learning_cnt=0, learning_data_file_path=in_file, industry_flag=0, zero_flag=0, add_x_data_volume=0, drop_train_percent=0, 
                                     x_interval_of_days=1, y_interval_of_days=1, prediction_y_info="UpDown", management_flag=1)


     
### [EOF]