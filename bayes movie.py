import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

dataPath = 'ml-1m/ratings.dat'
nUsers = 6040   # from the users.dat file TODO: read file for no. of entries
nMovies = 3706  # from the movies.dat file

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
    print('\n \u001b[4m\u001b[33mData Distribution\u001b[0m')
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

    prediction = clf.predict(xTest)
    print(f'Predicted probability:\n{clf.predict_proba(xTest)[:10]}')
    print(f'Prediction:\n{prediction[:10]}')
    print(f'Accuracy: {clf.score(xTest,yTest)*100:0.1f}%')

    # Compute confusion matrix
    print('\n \u001b[4m\u001b[33mConfusion Matrix\u001b[0m')
    print(confusion_matrix(yTest, prediction, labels=[0, 1]))
    print(f'precision: {precision_score(yTest, prediction, pos_label=1):0.2f}')
    print(f'recall: {recall_score(yTest, prediction, pos_label=1):0.2f}')
    print(f'f1 (pos) score: {f1_score(yTest, prediction, pos_label=1):0.2f}')
    print(f'f1 (neg) score: {f1_score(yTest, prediction, pos_label=0):0.2f}')