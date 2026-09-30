
import torch as tc
#import settings as s
import numpy as np
from copy import copy, deepcopy

class Checkpointer:
    def __init__(self):
        self.results= []
        self.epochs= []
        self.best_state_dict =  None
        self.epoch =0
        self.epoch_since_reset = 0
        self.best_epoch = 0
        self.best_epoch_since_reset = 0


    def update(self, result, state_dict):


        if self.epoch ==0:
            self.best_state_dict = deepcopy(state_dict)
            self.best_result = result

        #elif tc.all(result<tc.tensor(self.results, device = tc.device(result.device))): # this was s.device
        elif tc.all(result<tc.tensor(self.results)):

            self.best_state_dict = deepcopy(state_dict)
            self.best_epoch = copy(self.epoch)
            self.best_epoch_since_reset = copy(self.epoch_since_reset)
            self.best_result = result
        
        self.results.append(result)

        self.epochs_since_best_result_reset = self.epoch_since_reset - self.best_epoch_since_reset
        self.epochs_since_best_result = self.epoch - self.best_epoch

        self.epoch +=1
        self.epoch_since_reset +=1 

        return self.epochs_since_best_result_reset, self.epochs_since_best_result

        
    def reset(self):
        self.epoch_since_reset=0
        self.best_epoch_since_reset = 0
 


 
        
