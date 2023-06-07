#Imports
import gzip
import numpy as np
import matplotlib.pyplot as plt
from struct import unpack
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

def loadmnist(images, labels):

    # Get metadata for images
    images.read(4) # skip encoding data
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)
    N = labels.read(4)
    N = unpack('>I', N)[0]

    # Get data
    x = np.zeros((N, rows*cols), dtype=np.uint8)
    y = np.zeros(N, dtype=np.uint8)

    for i in range(N):
        for j in range(rows*cols):
            tmp_pixel = images.read(1)
            tmp_pixel = unpack('>B', tmp_pixel)[0]
            x[i][j] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]

    images.close()
    labels.close()
    return (x,y)


def plot_confusion_matrix(cm, title='Confusion Matrix', cmap='Pastel1'):
    plt.figure(figsize=(9, 9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=15)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], rotation=45, size=10)
    plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size=10)
    plt.tight_layout()
    plt.ylabel('Actual Label', size=15)
    plt.xlabel('Predicted Label', size=15)
    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x),
                         horizontalalignment='center',
                         verticalalignment='center')

#Set data as variables
TrainImages = gzip.open("C:/Users/epww/OneDrive/Documents/Work/MNIST data/train-images-idx3-ubyte.gz",'rb')
TestImages = gzip.open("C:/Users/epww/OneDrive/Documents/Work/MNIST data/t10k-images-idx3-ubyte.gz", 'rb')
TrainLabels = gzip.open("C:/Users/epww/OneDrive/Documents/Work/MNIST data/train-labels-idx1-ubyte.gz", 'rb')
TestLabels = gzip.open("C:/Users/epww/OneDrive/Documents/Work/MNIST data/t10k-labels-idx1-ubyte.gz", 'rb')

TrainImages, TrainLabels = loadmnist(TrainImages, TrainLabels)
TestImages, TestLabels = loadmnist(TestImages, TestLabels)

#Setup Logistic Regression
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(TrainImages, TrainLabels)
predictions = logisticRegr.predict(TestImages)
score = logisticRegr.score(TestImages, TestLabels)
print('Accuracy:', score)
confusion = metrics.confusion_matrix(TestLabels, predictions)
plt.figure()
plot_confusion_matrix(confusion)
plt.show()