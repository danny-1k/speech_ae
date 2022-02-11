import matplotlib.pyplot as plt

def save_loss_plot(train_loss,test_loss,f):
    
    plt.plot(train_loss,label='Train loss')
    plt.plot(test_loss,label='Test loss')

    plt.legend()

    plt.savefig(f)

    plt.close('all')