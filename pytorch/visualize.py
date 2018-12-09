import pickle
import matplotlib.pyplot as plt

def plot_progress(file_name='logs.pkl'):
    logs = pickle.load(open(file_name, 'rb'))

# (epoch, avg_epoch_loss, total_epoch_time, mrank, mAP)

    epoch = [e[0] for e in logs]
    avg_epoch_loss = [e[1] for e in logs]
    total_epoch_time = [e[2] for e in logs]
    mrank = [e[3] for e in logs]
    mAP = [e[4] for e in logs]

    plt.subplot(2, 2, 1)
    plt.plot(epoch, avg_epoch_loss, 'o-')
    plt.title('Loss v/s Epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(2, 2, 2)
    plt.plot(total_epoch_time, avg_epoch_loss, 'o-')
    plt.title('Loss v/s Time')
    plt.xlabel('time')
    plt.ylabel('loss')

    plt.subplot(2, 2, 3)
    plt.plot(epoch, mrank, 'o-')
    plt.title('Mean Rank v/s Epoch')
    plt.xlabel('epoch')
    plt.ylabel('rank')

    plt.subplot(2, 2, 4)
    plt.plot(epoch, mAP, 'o-')
    plt.title('Mean Average Precision v/s Epoch')
    plt.xlabel('epoch')
    plt.ylabel('MAP')

    plt.show()


if __name__ == '__main__':
    plot_progress(file_name='logs.pkl')
