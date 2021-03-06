import matplotlib.pyplot as plt
import numpy as np

def generate_loss_plot(task,data_dir='', step=32):
    loss = []
    f = open(data_dir + 'log_loss_' + task +'.txt', 'r', encoding='utf8')
    for line in f.readlines():
        loss.append(float(line))
    plt.figure(figsize=(13, 13))
    plt.title('Loss -  ' + task if task != 'total_loss' else 'Total Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot([(i + 1) * step for i in range(len(loss))], loss)
    plt.savefig(data_dir + 'loss_' + task + '.png')

def generate_error_plot(y_true, y_pred, data_dir=''):
    labels = ['Left eye', 'Right eye', 'Nose', 'Left mouth', 'Right mouth']
    rmse = {}
    i = 0
    p = 0
    while p < len(y_true[0]):
        rmse[labels[i]] = np.sqrt(np.mean(np.square(y_true[:, p:p + 2] - y_pred[:, p:p + 2])))
        p += 2
        i += 1
    plt.figure(figsize=(13, 13))
    fig, ax = plt.subplots()
    ax.barh(np.arange(len(labels)), np.array([rmse[key] for key in rmse]), align='center')
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('RMSE')
    ax.set_title('RMSE of each landmarks')
    fig.tight_layout()
    plt.savefig(data_dir + 'rmse_grouped_landmarks.png')

def generate_plot(data_dir='', mode='accuracy', label='', step=32):
    label = label.strip().lower()
    accuracy = []
    f = open(data_dir + 'log_' + label + '.txt', 'r', encoding='utf8')
    for line in f.readlines():
        accuracy.append(float(line) * 100) if mode =='accuracy' else accuracy.append(float(line))
    plt.figure(figsize=(13, 13))
    plt.title(mode.capitalize() + ' plot' + ' (' + label + ')') if mode == 'accuracy' else plt.title(mode.upper() + ' plot' + ' (' + label + ')')
    plt.xlabel('Iterations')
    plt.ylabel(mode.capitalize()) if mode == 'accuracy' else plt.ylabel(mode.upper())
    plt.plot([(i + 1) * step for i in range(len(accuracy))], accuracy)
    plt.savefig(data_dir + mode + '_' + label + '.png')

def generate_confusion_matrix_plot(conf_mat, classes, normalize=False, title='', data_dir=''):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    if normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(13, 13))
    fig, ax = plt.subplots()
    im = ax.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(conf_mat.shape[1]),
           yticks=np.arange(conf_mat.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True labels',
           xlabel='Predicted labels')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = conf_mat.max() / 2.
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(j, i, format(conf_mat[i, j], fmt),
                    ha="center", va="center",
                    color="white" if conf_mat[i, j] > thresh else "black")
    fig.tight_layout()
    if normalize:
        plt.savefig(data_dir + 'confusion matrix normalized.png')
    else:
        plt.savefig(data_dir + 'confusion matrix.png')


