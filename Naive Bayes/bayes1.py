import numpy as np
from collections import defaultdict

def get_label_indeces(labels):
    '''
    Group samples based on their labels and return indeces
    @param labels: list of labels
    @return: dict, {class1: [indeces], class2: [indices]}
    '''
    label_indices = defaultdict(list)
    for index, label in enumerate(labels):
        label_indices[label].append(index)
    return label_indices

if __name__ == '__main__':
    # X training array
    xTrain = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
        [1, 1, 0]])
    
    # Y training vector
    yTrain = ['Y', 'N', 'Y', 'Y']

    # test array
    xTest = np.array([[1, 1, 0]])

    labelIdx = get_label_indeces(yTrain)
    print(labelIdx)
