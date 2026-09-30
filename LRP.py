from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch as tc
import os
from tqdm import tqdm
import settings as s

def compute_LRP_scores(model, ds, label_pos, fold, device):
    print('computing LRP scores')

    LRP_PATH = './results/'+s.mode+'/LRP/'

    if not os.path.exists(LRP_PATH):
        os.makedirs(LRP_PATH)

    device = tc.device(device)
    model.eval().to(device)


    therapy, diagnostics = ds.therapy.to(device), ds.diagnostics.to(device)

    pred = model.forward(therapy, diagnostics)
    
    all_labels=[]

    R = tc.zeros_like(pred)
    R[:,label_pos] = pred[:,label_pos].clone()

    relevances = model.relprop(R)        

    all_samples = []
    for sample in tqdm(range(relevances.shape[0])):

        sample_interaction_matrix = relevances[sample,:,:]
            
        interaction_frame = pd.DataFrame(sample_interaction_matrix.cpu().detach().numpy(), index = ds.ther_features, columns = ds.diag_features)
        interaction_frame['therapy'] = interaction_frame.index
        interaction_frame_long = pd.melt(interaction_frame, id_vars = 'therapy', var_name = 'diagnostics', value_name = 'BiLRP_raw')


        #add input values
        therapy_inputs = pd.DataFrame({'therapy_inputs': therapy[sample,:].cpu().numpy(), 'therapy': ds.ther_features})
        diagnostics_inputs = pd.DataFrame({'diagnostics_inputs': diagnostics[sample,:].cpu().numpy(), 'diagnostics': ds.diag_features})
        interaction_long_with_inputs = interaction_frame_long.merge(therapy_inputs, how='left').merge(diagnostics_inputs, how = 'left')


        # reverse feature expansion
        interaction_frame_long_ant = interaction_long_with_inputs[~interaction_frame_long['diagnostics'].str.endswith('_rev')] 
        interaction_frame_long_ant = interaction_frame_long_ant.rename(columns = {'BiLRP_raw': 'BiLRP_ant', 'therapy_inputs': 'therapy_inputs', 'diagnostics_inputs' : 'diagnostics_inputs_ant'})
           
        interaction_frame_long_rev = interaction_long_with_inputs[interaction_frame_long['diagnostics'].str.endswith('_rev')]
        interaction_frame_long_rev= interaction_frame_long_rev.rename(columns = {'BiLRP_raw': 'BiLRP_rev', 'therapy_inputs': 'therapy_inputs', 'diagnostics_inputs':'diagnostics_inputs_rev'})
        interaction_frame_long_rev['diagnostics'] = interaction_frame_long_rev['diagnostics'].str.split('_rev', expand=True)[0]

        interaction_frame_merged = interaction_frame_long_ant.merge(interaction_frame_long_rev, how='left')

        interaction_frame_merged['BiLRP'] = interaction_frame_merged[['BiLRP_ant', 'BiLRP_rev']].sum(axis=1, skipna=True)
        interaction_frame_merged['diagnostics_inputs'] = interaction_frame_merged[['diagnostics_inputs_ant', 'diagnostics_inputs_rev']].sum(axis=1, skipna=True)


        #add additional information
        interaction_frame_merged['sample_name'] = ds.sample_names[sample]
        interaction_frame_merged['prediction_label'] = ds.labs_features[label_pos]
        interaction_frame_merged['prediction_value'] = pred[sample,label_pos].clone().cpu().detach().numpy()

        if s.only_select_given_therapies:
            interaction_frame_merged = interaction_frame_merged[interaction_frame_merged["therapy_inputs"]!=0]

        all_samples.append(interaction_frame_merged)
            
    all_frame=pd.concat(all_samples, axis=0)
    all_frame['fold'] = fold

    all_frame.to_csv(LRP_PATH + '/LRP_results_label' +  ds.labs_features[label_pos]+ 'fold' + str(fold) + '.csv')






     

    
