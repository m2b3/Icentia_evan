import os
import gzip
import pickle
import numpy as np
import pandas as pd

def readSubject(subjectID, dataFolder): 
    with gzip.open(f"{dataFolder}/{subjectID:05d}_batched.pkl.gz", "rb") as file: 
        ECGTotal = np.array(pickle.load(file))
    with gzip.open(f"{dataFolder}/{subjectID:05d}_batched_lbls.pkl.gz", "rb") as file: 
        peakDictTotal = np.array(pickle.load(file))

    return ECGTotal, peakDictTotal

