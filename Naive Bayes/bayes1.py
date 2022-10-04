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

def get_likelihood(features, label_indices, smoothing = 0):
    '''
    Compute likelihood based on training samples

    @param features: matrix of features
    @param label_inices: grouped sample indices by class
    @param smoothing: integer, additive smoothing parameter
    @return: dictonary, with class as key, corresponding conditional
             probability P(feature | class) vector as value
    '''
    likelihood = {}

    for label, indices in label_indices.items():
        likelihood[label] = features[indices, :].sum(axis = 0) + smoothing
        total_count = len(indices)
        likelihood[label] = likelihood[label] / (total_count + 2 * smoothing)
    
    return likelihood

def get_posterior(X, prior, likelihood):
    '''
    Compute posterior of testing samples, based on prior and likelihood

    @param X: test samples
    @param prior: dictionary, with class label as key, corresponding prior
                  as the value
    @param likelihood: dictionary, with class label as key, corresponding
                       conditional probability vector as value
    @return: dictonary, with class label as key, corresponding posterior as
             value
    '''
    posteriors = []

    for x in X:
        # posterior is proportional to prior × likelihood
        posterior = prior.copy()
        for label, likelihood_label in likelihood.items():
            for index, bool_value in enumerate(x):
                posterior[label] *= likelihood_label[index] if \
                    bool_value else (1 - likelihood_label[index])
        
        # normalize data
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        
        posteriors.append(posterior.copy())

    return posteriors

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

    likelihood = get_likelihood(xTrain, labelIdx, smoothing = 1)
    print(f'Likelihood:\n{likelihood}')

    posterior = get_posterior(xTest, prior, likelihood)
    print(f'Posterior:\n{posterior}')