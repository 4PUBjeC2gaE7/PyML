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

def get_prior(label_indices):
    '''
    Compute prior based on training samples
    @param label_indices: gropued sample indices by class
    @return: dictionary, with class label as key, corresponding prior as
             the value
    '''
    prior = {label: len(indices) for label, indices in label_indices.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= total_count
    return prior

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

    prior = get_prior(labelIdx)
    print(f'Prior: {prior}')