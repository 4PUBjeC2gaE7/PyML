from re import T
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, precision_score, roc_auc_score, \
                            recall_score, f1_score, classification_report

dataPath = 'ml-1m/ratings.dat'
nUsers = 6040   # from the users.dat file TODO: read file for no. of entries
nMovies = 3706  # from the movies.dat file

def printHead(myString):
    if type(myString) is str:
        print(f'\n \u001b[4m\u001b[33m{myString}\u001b[0m')


def load_rating_data(dataPath, nUsers, nMovies):
    '''
    Load rating data from file and also return the number of ratings for each
    movie and movieId index mapping

    @param dataPath: path of the rating datafile
    @param nUsers: number of users
    @param nMovies: number of movies that have ratings
    @return: rating data in the numpy array of [user, movie];
             movieNRating, {movieId: number or ratings};
             movieIdMapping, {movieId: column index in rating data}
    '''
    # data is 2D array mapping users to movies and ratings
    # unranked data is given a zero
    data = np.zeros([nUsers, nMovies], dtype = np.float32)
    movieIdMapping = {}
    movieNRating = defaultdict(int)

    with open(dataPath, 'r') as file:
        for line in file.readlines()[1:]:
            # read line
            userId, movieId, rating, _ = line.split('::')
            # parse data
            userId = int(userId) - 1
            rating = int(rating)
            # add movie to mapping
            if movieId not in movieIdMapping:
                movieIdMapping[movieId] = len(movieIdMapping)
            # add data to array
            data[userId, movieIdMapping[movieId]] = rating
            if rating > 0:
                movieNRating[movieId] += 1
    return data, movieNRating, movieIdMapping

def display_distribution(data):
    '''
    Prints the distribution of ratings from zero (unrated) to 5

    @param data: rating data in the numpy array of [user, movie];
    '''
    values, counts = np.unique(data, return_counts = True)
    printHead('Data Distribution')
    for value, count in zip(values, counts):
        print(f' rating {int(value)}:{count:-8d}')
    print('')

if __name__ == '__main__':
    recommended = 3
    data, movieNRating, movieIdMapping = load_rating_data(dataPath, nUsers, nMovies)

    display_distribution(data)

    movieIdMost, nRatingsMost = sorted(movieNRating.items(),
                                       key = lambda d: d[1], reverse = True)[0]
    print(f'Movie ID {movieIdMost} has {nRatingsMost} ratings!')
    # Build X-Y dataset
    xRaw = np.delete(data, movieIdMapping[movieIdMost], axis = 1)
    yRaw = data[:, movieIdMapping[movieIdMost]]
    # Discard samples without a rating for "Most" Movie
    X = xRaw[yRaw > 0]
    Y = yRaw[yRaw > 0]
    print(f'Shape of X: {X.shape}')
    print(f'Shape of Y: {Y.shape}')
    display_distribution(Y)
    # Check for recommendation
    Y[Y <= recommended] = 0
    Y[Y > recommended] = 1
    nPos = (Y == 1).sum()
    nNeg = (Y == 0).sum()
    print(f'There are {nPos} positive samples and {nNeg} negative samples')

    # Train sklearn model
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(f'Training: {len(yTrain)}\tTest: {len(yTest)}')

    # Train multinomial Naive Bayes
    clf = MultinomialNB(alpha=1.0, fit_prior=True)
    clf.fit(xTrain, yTrain)

    predProb = clf.predict_proba(xTest)
    prediction = clf.predict(xTest)
    print(f'Predicted probability:\n{predProb[:10]}')
    print(f'Prediction:\n{prediction[:10]}')
    print(f'Accuracy: {clf.score(xTest,yTest)*100:0.1f}%')

    # Compute confusion matrix
    printHead('Confusion Matrix')
    print(confusion_matrix(yTest, prediction, labels=[0, 1]))
    print(f'\nprecision: {precision_score(yTest, prediction, pos_label=1):0.2f}')
    print(f'recall: {recall_score(yTest, prediction, pos_label=1):0.2f};  a.k.a TPR')
    print(f'f1 (neg) score: {f1_score(yTest, prediction, pos_label=0):0.2f}')
    print(f'f1 (pos) score: {f1_score(yTest, prediction, pos_label=1):0.2f}')
    
    printHead('Classification Report')
    report = classification_report(yTest, prediction)
    print(report)

    # Messing with "Receiver Operating Characteristic (ROC)"
    posProb = predProb[:,1]
    thresholds = np.arange(0.0, 1.1, 0.05)
    truePos, falsePos = [0]*len(thresholds), [0]*len(thresholds)
    for pred, y in zip(posProb, yTest):
        for i, threshold in enumerate(thresholds):
            if pred >= threshold:
                # if truth and predict are both '1'
                if y == 1:
                    truePos[i] += 1
                # if truth is '0' and predict is '1'
                else:
                    falsePos[i] += 1
            else:
                break

    nPosTest = (yTest == 1).sum()
    nNegTest = (yTest == 0).sum()
    truePosRate = [tp / nPosTest for tp in truePos]
    falsePosRate = [fp / nNegTest for fp in falsePos]
    # Calculate Area Under the Curve (AUC) score
    print(f'AUC Score: {roc_auc_score(yTest, posProb):0.2f}')

    plt.figure()
    lw = 2
    plt.plot(falsePosRate, truePosRate,
             color = 'darkorange', lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()