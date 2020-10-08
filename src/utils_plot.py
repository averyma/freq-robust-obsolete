import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
    
def plot_standard_adv(log):
    train_acc = np.array(log["train_acc_ep"])[:,2]
    test_acc = np.array(log["test_acc"])[:,2]
    adv_acc_curr = np.array(log["pgd20_acc_curr"])[:,2]
    adv_acc_prev = np.array(log["pgd20_acc_prev"])[:,2]
    
    train_loss= np.array(log["train_loss_ep"])[:,2]
    test_loss = np.array(log["test_loss"])[:,2]
    adv_loss_curr = np.array(log["pgd20_loss_curr"])[:,2]
    adv_loss_prev = np.array(log["pgd20_loss_prev"])[:,2]
        
    fig = plt.figure(figsize = [15,7])
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1,2)
    plot_list =list(range(1,train_loss.shape[0]+1))
    
    fig.add_subplot(gs[0,0]).plot(plot_list, train_acc, "C1", label = "train", linewidth=5.0, marker = "")
    fig.add_subplot(gs[0,0]).plot(plot_list, test_acc, "C2", label = "test", linewidth=5.0, marker = "")
    fig.add_subplot(gs[0,0]).plot(plot_list, adv_acc_curr, "C3", label = "pgd20_curr", linewidth=5.0, marker = "")
    fig.add_subplot(gs[0,0]).plot(plot_list, adv_acc_prev, "C4", label = "pgd20_prev", linewidth=5.0, marker = "")
    
    fig.add_subplot(gs[0,1]).plot(plot_list, train_loss, "C1", label = "train", linewidth=5.0, marker = "")
    fig.add_subplot(gs[0,1]).plot(plot_list, test_loss, "C2", label = "test", linewidth=5.0, marker = "")
    fig.add_subplot(gs[0,1]).plot(plot_list, adv_loss_curr, "C3", label = "pgd20_curr", linewidth=5.0, marker = "")
    fig.add_subplot(gs[0,1]).plot(plot_list, adv_loss_prev, "C4", label = "pgd20_prev", linewidth=5.0, marker = "")

    
    fig.add_subplot(gs[0,0]).set_xlabel("epochs", fontsize = 25)
    fig.add_subplot(gs[0,1]).set_xlabel("epochs", fontsize = 25)
    fig.add_subplot(gs[0,0]).set_title("Accuracy", fontsize = 25)
    fig.add_subplot(gs[0,1]).set_title("Loss", fontsize = 25)


    fig.add_subplot(gs[0,0]).grid()
    fig.add_subplot(gs[0,1]).grid()

    fig.add_subplot(gs[0,0]).legend(prop={"size": 20})
    fig.add_subplot(gs[0,1]).legend(prop={"size": 20}, loc = "upper left")

    fig.add_subplot(gs[0,0]).tick_params(labelsize=20)
    fig.add_subplot(gs[0,1]).tick_params(labelsize=20)

    fig.add_subplot(gs[0,0]).set_ylim([0,100])
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    fig.tight_layout()
    return fig

