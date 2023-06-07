def plot_confusion_matrix(cm, title='Confusion Matrix', cmap='Pastel1'):
    plt.figure(figsize=(9,9))
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

confusion = metrics.confusion_matrix(TestLabels, predictions)
print('Confusion Matrix')
print(confusion)
plt.figure()
plot_confusion_matrix(confusion)
plt.show()
