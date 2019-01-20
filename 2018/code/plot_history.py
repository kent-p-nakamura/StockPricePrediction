"""
plot_history.py: 学習結果のacc/lossを画像として保存
"""
import matplotlib.pyplot as plt

def plot_history(hist, path, num, index, histName):
    
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(15,6))
    
    # plot
    axL.plot(hist.history['acc'], linestyle='-', color='b', label='train')
    axL.plot(hist.history['val_acc'], linestyle='-', color='#ff4500', label='test')
    # x axis
    axL.set_xlabel('epochs')
    
    # y axis
    axL.set_ylabel('acc')
    
    # legend and title
    axL.legend(loc='best')
    axL.set_title("Accuracy_{0}_{1}_{2}".format(histName, num, index))
    
    ################
    # plot
    axR.plot(hist.history['loss'], linestyle='-', color='b', label='train')
    axR.plot(hist.history['val_loss'], linestyle='-', color='#ff4500', label='test')
    
    # x axis
    axR.set_xlabel('epochs')
    
    # y axis
    axR.set_ylabel('loss')
    
    # legend and title
    axR.legend(loc='best')
    axR.set_title("Loss_{0}_{1}_{2}".format(histName, num, index))
    
    
    # save as png
    plt.savefig(path+"history_{0}_{1}_{2}".format(histName, num, index))
    #plt.show()
    plt.close()
    
    
def plot_history_acc(hist, path, num):
    # plot
    plt.plot(hist.history['acc'], linestyle='-', color='b', label='train')
    plt.plot(hist.history['val_acc'], linestyle='-', color='#ff4500', label='test')
    
    # x axis
    plt.set_xlabel('epochs')
    
    # y axis
    plt.set_ylabel('acc')
    
    # legend and title
    plt.legend(loc='best')
    plt.set_title("Accuracy_{}".format(num))
    
    
    # save as png
    plt.savefig(path+"history_acc_{}.png".format(num))
    plt.show()
    plt.close()


def plot_history_loss(hist, path, num):
   # plot
    plt.plot(hist.history['loss'], linestyle='-', color='b', label='train')
    plt.plot(hist.history['val_loss'], linestyle='-', color='#ff4500', label='test')
    
    # x axis
    plt.set_xlabel('epochs')
    
    # y axis
    plt.set_ylabel('loss')
    
    # legend and title
    plt.legend(loc='best')
    plt.set_title("Loss_{}".format(num))
    
    
    # save as png
    plt.savefig(path+"history_loss_{}.png".format(num))
    plt.show()
    plt.close()



if __name__ == '__main__':
    print("please use this file(plot_history.py) in other file")
    

