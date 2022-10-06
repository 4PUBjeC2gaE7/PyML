import numpy as np
from collections import defaultdict

dataPath = 'ml-1m/ratings.dat'
nUsers = 6040   # from the users.dat file TODO: read file for no. of entries
nMovies = 3706  # from the movies.dat file

def load_rating_data(dataPath, nUsers, nMovies):
    '''
    Load rating data from file and also return the number of ratings for each movie and movieId index mapping

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