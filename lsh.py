import pandas as pd
import numpy as np


class LSH():
    def __init__(self):
        self._lsh = None
        self._list_of_seeds = np.array('init_elem')
        
        
        
    def add_seeds(self, seeds):
        for seed in seeds:
            self._list_of_seeds.append(seeds)
            
            
    def random_hyperplanes(self):
        pass