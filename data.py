import torch as tc
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import settings as s
import pandas as pd
from sklearn.preprocessing import StandardScaler,PowerTransformer

location = s.location

label_cutoff = None

if location == 'M':
    class MyScaler:
        def __init__(self):
            pass

        def fit(self, data):
            pass

        def transform(self,data):
            return data

        def fit_transform(self, data):
            return data
        
    
    diagnostics = pd.DataFrame(np.array(tc.randn(4000,10)), columns = ['a', 'b', 'c', 'd', 'e', 'f', 'a_rev', 'b_rev', 'c_rev', 'd_rev'])
    
    therapy = pd.DataFrame(np.array(tc.randn(4000,5)), columns = ['v', 'w', 'x','y', 'z'])

    labels0 = pd.DataFrame((np.array(diagnostics)[:,0] * np.array(therapy)[:,0])[:,None]) 
    labels1 = pd.DataFrame((np.array(diagnostics)[:,1] * np.array(therapy)[:,1])[:,None])
    labels = pd.concat((labels0, labels1), axis=1)

    labels = labels + 0.2* np.array(tc.rand_like(tc.tensor(np.array(labels)))) 
    labels[np.array(tc.rand_like(tc.tensor(np.array(labels)))>0.9)] = np.nan
    labels.columns = ['label0', 'label1']



    def compute_grade_labels(grade):
        """Derive resampling frequencies and complication flags for an adverse-event grade.

        The regression targets (nadir lab values in `labels`) are identical for every
        grade; the grade only defines how extreme a nadir must be to count as an event.
        Samples whose nadir falls below the grade quantile are the (rare) adverse events;
        each sample's frequency is the size of its group, so training's 1/frequency weight
        oversamples the rare group. A lower quantile (higher grade) => rarer => stronger
        oversampling. Mirrors the real pipeline where `label_counts_*_<grade>` and
        `complication_yesno_*_<grade>` are loaded per grade."""
        quantile = s.grade_event_quantile[grade]
        freq_cols, comp_cols = {}, {}
        for col in labels.columns:
            vals = labels[col]
            observed = vals.notna()
            threshold = vals.quantile(quantile)
            is_event = (vals < threshold) & observed          # rare adverse event of this grade
            event_count = int(is_event.sum())
            nonevent_count = int((~is_event & observed).sum())

            freq = pd.Series(np.nan, index=labels.index)
            freq[is_event] = event_count
            freq[~is_event & observed] = nonevent_count

            comp = pd.Series(np.nan, index=labels.index)
            comp[observed] = is_event[observed].astype(float)

            freq_cols[col] = freq
            comp_cols[col] = comp
        return pd.DataFrame(freq_cols), pd.DataFrame(comp_cols)

    # default grade for module-level use / standalone imports
    labels_frequencies, label_complications = compute_grade_labels(s.grades[0])

    sex = pd.DataFrame({'sex':((tc.rand(4000)>0.5) *1.0).numpy()})



class DS_Interaction(Dataset):
    def __init__(self, diag, ther, labs, labs_frequencies = None, labs_complications = None):

        self.diag_features = np.array(diag.columns)
        self.ther_features = np.array(ther.columns)
        self.labs_features = np.array(labs.columns)

        self.sample_names = np.array(diag.index)
        assert all(diag.index == ther.index), 'sample names do not fit'
        assert all(diag.index == labs.index), 'sample names do not fit'
        assert all(labs.columns == labs_frequencies.columns), 'lab names wrong'
        assert all(labs.index == labs_frequencies.index), 'lab names wrong'
       
        self.diagnostics = tc.tensor(np.array(diag)).float()
        self.therapy = tc.tensor(np.array(ther)).float()
        self.labels = tc.tensor(np.array(labs)).float()
        
        self.label_sample_frequencies = tc.from_numpy(np.array(labs_frequencies))
        self.label_iscomplication = tc.from_numpy(np.array(labs_complications).astype(float))


        self.n_samples, self.n_diagnostics_features = self.diagnostics.shape
        self.n_therapy_features = self.therapy.shape[1]
        self.n_labels = self.labels.shape[1]

        assert self.diagnostics.shape[0] == self.therapy.shape[0], 'somethings wrong'
        assert self.diagnostics.shape[0] == self.labels.shape[0], 'somethings wrong'


    def __len__(self):
        return self.diagnostics.shape[0]

    def __getitem__(self, idx):
        return self.therapy[idx,:], self.diagnostics[idx,:], self.labels[idx,:], self.label_iscomplication[idx,:], idx
        

class ID_generator:
    def __init__(self, n_splits, n_samples):
        self.n_splits = n_splits

        self.n_samples = n_samples

        tc.manual_seed(0)
        self.all_ids = tc.randperm(n_samples)

        self.all_test_splits = np.array_split(self.all_ids, n_splits)

        
    def get_ids(self, current_split):
        test_val_ids = self.all_test_splits[current_split]
        tv_size = test_val_ids.shape[0]
        test_ids, val_ids = test_val_ids[:tv_size//2], test_val_ids[tv_size//2:]

        train_ids = np.setdiff1d(self.all_ids, test_val_ids, assume_unique=True) 

        return np.array(train_ids), np.array(val_ids), np.array(test_ids)

#new version in which all data ends up in test sets
class ID_generator:
    def __init__(self, n_splits, n_samples):
        self.n_splits = n_splits

        self.n_samples = n_samples

        tc.manual_seed(0)
        self.all_ids = tc.randperm(n_samples)

        self.all_test_splits = np.array_split(self.all_ids, n_splits)

        
    def get_ids(self, current_split):
        test_ids = self.all_test_splits[current_split]
        t_size = test_ids.shape[0]

        train_val_ids = np.setdiff1d(self.all_ids, test_ids, assume_unique=True) 

        #shuffle train val set
        tc.manual_seed(current_split)
        shuffled_train_val_ids = train_val_ids[tc.randperm(train_val_ids.shape[0])]
        train_val_ids_split =  np.array_split(shuffled_train_val_ids, self.n_splits-1)

        val_ids = train_val_ids_split[0]

        train_ids =  np.setdiff1d(train_val_ids, val_ids, assume_unique=True) 
        print('data leakage:', np.intersect1d(train_ids, val_ids), np.intersect1d(train_ids, test_ids), np.intersect1d(val_ids, test_ids))

        return np.array(train_ids), np.array(val_ids), np.array(test_ids)


    


    




