import torch as tc
from torch.optim import Adam, SGD
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
from copy import deepcopy
from tqdm import tqdm
from torch.utils.data.sampler import WeightedRandomSampler
import settings as s
from utils import Checkpointer
from torch.nn.utils import clip_grad_norm_
import gc
   



def test_model(model, ds_test, label_pos, fold, label_scaler, device):

    device = tc.device(device)

    dl_test = DataLoader(ds_test, batch_size= ds_test.n_samples)

    model.eval().to(device)

    for therapy_test, diagnostics_test, labels_test, label_complication_test, idx in dl_test:
        
        therapy_test, diagnostics_test, labels_test, label_complication_test = therapy_test.to(device), diagnostics_test.to(device), labels_test.to(device), label_complication_test.to(device)

        pred_test_regression = model.regress(therapy_test, diagnostics_test)

        rescaled_prediction = label_scaler.inverse_transform(pd.DataFrame(pred_test_regression.detach().cpu().numpy(), columns = ds_test.labs_features))
        rescaled_labels= label_scaler.inverse_transform(pd.DataFrame(labels_test.detach().cpu().numpy(), columns = ds_test.labs_features))

        prediction_frame = pd.DataFrame({'prediction':pred_test_regression[:,label_pos].detach().cpu().numpy().squeeze(),
                                        'rescaled_prediction': rescaled_prediction[:,label_pos]})
        
        prediction_frame['label'] =  np.array(ds_test.labs_features)[label_pos]
        prediction_frame['sample_name'] = ds_test.sample_names[idx].squeeze()
        prediction_frame_long = prediction_frame#.melt(id_vars = 'sample_name', var_name = 'label', value_name = 'prediction')

        ground_truth_frame = pd.DataFrame({'ground_truth':labels_test[:,label_pos].detach().cpu().numpy().squeeze(),
                                            'rescaled_ground_truth': rescaled_labels[:, label_pos],
                                            'complication_ground_truth':label_complication_test[:,label_pos].detach().cpu().numpy().squeeze()})
        
        ground_truth_frame['label'] = np.array(ds_test.labs_features)[label_pos]
        ground_truth_frame['sample_name'] = ds_test.sample_names[idx].squeeze()
        ground_truth_frame_long = ground_truth_frame#.melt(id_vars = 'sample_name', var_name = 'label', value_name = 'ground_truth')

        test_df = ground_truth_frame_long.merge(prediction_frame_long, how='left')
        test_df['fold'] = fold


    test_df.to_csv('./results/'+s.mode  +'/testing/training_loss_one_label_'+str(np.array(ds_test.labs_features)[label_pos])+'_fold_' + str(fold) + '.csv')





def train_test_one_label_overall(model, ds_train, ds_val, ds_test, label_pos, batch_size, n_epochs, fold, device):

    device = tc.device(device)
    

    if s.resampling:
        #print('resampling...label frequencies: ', (ds_train.label_sample_frequencies[:,label_pos].unique() / ds_train.label_sample_frequencies[:,label_pos].unique().sum())[1:])

        probs = tc.where(tc.isnan(ds_train.label_sample_frequencies[:,label_pos]), 0.0, 1/ds_train.label_sample_frequencies[:,label_pos])*1e4
        print(probs)
        sampler = WeightedRandomSampler(probs, ds_train.label_sample_frequencies.shape[0])
        dl_train = DataLoader(ds_train, batch_size=batch_size, sampler = sampler)

    else: 
        #dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

        probs = tc.where(tc.isnan(ds_train.label_sample_frequencies[:,label_pos]), 0.0, 1)*1e4
        sampler = WeightedRandomSampler(probs, ds_train.label_sample_frequencies.shape[0])
        dl_train = DataLoader(ds_train, batch_size=batch_size, sampler = sampler)
        print('no resampling')

    dl_val = DataLoader(ds_val, batch_size= ds_val.n_samples)


    regression_criterion = nn.HuberLoss()
    #regression_criterion = nn.MSELoss()

    checkpointer = Checkpointer()


    for current_round, lr in enumerate([s.starting_lr, s.starting_lr]):
        #optimizer = Adam(model.parameters(), lr = lr*1e-2, weight_decay=1e-4) #was 1e-6
        dl_train = DataLoader(ds_train, batch_size=batch_size*(2**current_round), sampler = sampler)
        optimizer = tc.optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay=1e-3) #1e-4


        for epoch in range(n_epochs):
            tc.cuda.empty_cache()
            gc.collect()


            model.train().to(device)
            for therapy_train, diagnostics_train, labels_train, label_complication, _ in dl_train:
                therapy_train, diagnostics_train, labels_train, label_complication = therapy_train.to(device), diagnostics_train.to(device), labels_train.to(device), label_complication.to(device)

                optimizer.zero_grad()

                if s.regression_loss:
                    pred_train = model.regress(therapy_train, diagnostics_train)

                    if s.fit_all_labels:
                        pred_train_flat = pred_train.flatten()
                        labels_train_flat = labels_train.flatten()
                        mask = ~tc.isnan(labels_train_flat)
                        loss_train = regression_criterion(pred_train_flat[mask], labels_train_flat[mask])
        
                    else:
                        mask= ~tc.isnan(labels_train[:, label_pos])
                        loss_train = regression_criterion(pred_train[mask,label_pos], labels_train[mask,label_pos])
                    
                    loss_train.backward()
                    clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()


     
           #testing
            model.eval().to(device)
            loss_list = []
            for therapy_val, diagnostics_val, labels_val, label_complication_val, idx in dl_val:

                

                therapy_val, diagnostics_val, labels_val, label_complication_val = therapy_val.to(device), diagnostics_val.to(device), labels_val.to(device), label_complication_val.to(device)

                pred_val_regression = model.regress(therapy_val, diagnostics_val)
                mask_val= ~tc.isnan(labels_val[:, label_pos])
                val_loss_regression = regression_criterion(pred_val_regression[mask_val,label_pos], labels_val[mask_val,label_pos])

                val_loss = val_loss_regression
                epochs_since_best_result_reset, epochs_since_best_result = checkpointer.update(result = val_loss.item(), state_dict = model.state_dict())
                star = '*' if epochs_since_best_result == 0 else ''

                print('epoch:', epoch, '| fold:', fold,'| valloss:', str(np.round(np.array(val_loss.cpu().detach()),4)) + star, '| lr:', lr )

            if epochs_since_best_result_reset >=s.epochs_until_restart:
                model.load_state_dict(checkpointer.best_state_dict)
                checkpointer.reset()
                break
               

    model.load_state_dict(checkpointer.best_state_dict)

    print('best validation loss:', checkpointer.best_result)
    prediction_frame = pd.DataFrame({'prediction':pred_val_regression[:,label_pos].detach().cpu().numpy().squeeze()})
    prediction_frame['label'] =  np.array(ds_val.labs_features)[label_pos]
    prediction_frame['sample_name'] = ds_val.sample_names[idx].squeeze()
    prediction_frame_long = prediction_frame#.melt(id_vars = 'sample_name', var_name = 'label', value_name = 'prediction')

    ground_truth_frame = pd.DataFrame({'ground_truth':labels_val[:,label_pos].detach().cpu().numpy().squeeze()})
    ground_truth_frame['label'] = np.array(ds_val.labs_features)[label_pos]
    ground_truth_frame['sample_name'] = ds_val.sample_names[idx].squeeze()
    ground_truth_frame_long = ground_truth_frame#.melt(id_vars = 'sample_name', var_name = 'label', value_name = 'ground_truth')

    val_df = ground_truth_frame_long.merge(prediction_frame_long, how='left')

    val_df.to_csv('./results/'+s.mode+'/training/training_loss_one_label_'+str(np.array(ds_val.labs_features)[label_pos])+'_fold_' + str(fold) + '.csv')

    
    if not os.path.exists('./results/'+s.mode+'/models/fold' + str(fold) + '/'):
        os.makedirs('./results/'+s.mode+'/models/fold' + str(fold) + '/')
    tc.save(model.state_dict(), './results/'+s.mode+'/models/fold' + str(fold) + '/label'+ str(label_pos) + '.pt')

       




