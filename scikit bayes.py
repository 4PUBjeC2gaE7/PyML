if __name__ == '__main__':
    import numpy as np
    from sklearn.naive_bayes import BernoulliNB

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

    clf = BernoulliNB(alpha=1.0, fit_prior=True)
    clf.fit(xTrain, yTrain)

    predProb = clf.predict_proba(xTest)
    print(f'Predicted probabilities:\n{predProb}')

    pred = clf.predict(xTest)
    print(f'Prediction: {pred}')