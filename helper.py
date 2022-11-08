import numpy as np
import pandas as pd
movies = pd.read_csv("./data/movies.csv")

def joinMovieNames(arg: pd.DataFrame):
    return (arg.set_index('movie_id').join(movies.set_index('movieId'))).drop(columns = ["genres"])

    
'''
NN()
'''