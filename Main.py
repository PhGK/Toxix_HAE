from NN import Model, Interaction_Model, Simple_Model
import os
from training import train_test_one_label_overall, test_model
from torch.utils.data import DataLoader
from LRP import compute_LRP_scores
from data import diagnostics, therapy, labels, ID_generator, DS_Interaction, MyScaler, compute_grade_labels
import settings as s
from sklearn.preprocessing import PowerTransformer, MaxAbsScaler
import pandas as pd
import numpy as np
import gc
import torch as tc

#before vacation: made first part until interaction linear!
#compare to results

BASE_MODE = s.mode


def run_grade(grade):
    # grade only changes the resampling frequencies / complication flags, not the
    # regression targets. Results for this grade go to results/<BASE_MODE>_grade<grade>/.
    labels_frequencies, label_complications = compute_grade_labels(grade)

    for path in ['./results/' + s.mode +'/data/', './results/' + s.mode +'/training/', './results/' + s.mode +'/testing/']:
        if not os.path.exists(path):
            os.makedirs(path)


    print(diagnostics.shape, therapy.shape, labels.shape, labels_frequencies.shape)
    assert all(diagnostics.index == therapy.index), 'sample names do not fit'
    assert all(diagnostics.index == labels.index), 'sample names do not fit'
    assert all(labels.index == labels_frequencies.index), 'sample names do not fit'
    assert all(labels.columns == labels_frequencies.columns), 'sample names do not fit'
    assert all(labels.columns == label_complications.columns), 'sample names do not fit'



    id_generator = ID_generator(n_splits = s.n_splits, n_samples = diagnostics.shape[0])

    for split in range(s.n_splits):
        tc.cuda.empty_cache()
        gc.collect()

        train_ids, val_ids, test_ids = id_generator.get_ids(current_split = split)

        scaler = MyScaler()
        scaled_train_diagnostics = scaler.fit_transform(diagnostics.iloc[train_ids,:])
        scaled_val_diagnostics = scaler.transform(diagnostics.iloc[val_ids,:])
        scaled_test_diagnostics = scaler.transform(diagnostics.iloc[test_ids,:])

        train_labels =  labels.iloc[train_ids,:]
        val_labels =  labels.iloc[val_ids,:]
        test_labels =  labels.iloc[test_ids,:]


        label_scaler = PowerTransformer()
        scaled_train_labels = pd.DataFrame(label_scaler.fit_transform(train_labels), index = train_labels.index, columns = train_labels.columns)
        scaled_val_labels = pd.DataFrame(label_scaler.transform(val_labels), index = val_labels.index, columns = val_labels.columns)
        scaled_test_labels = pd.DataFrame(label_scaler.transform(test_labels), index = test_labels.index, columns = test_labels.columns)

        therapy_scaling = therapy.iloc[train_ids,:].max()
        scaled_train_therapy = therapy.iloc[train_ids,:] / therapy_scaling
        scaled_val_therapy = therapy.iloc[val_ids,:] / therapy_scaling
        scaled_test_therapy = therapy.iloc[test_ids,:] / therapy_scaling
        #train_therapy = therapy.iloc[train_ids,:]
        #val_therapy = therapy.iloc[val_ids,:]
        #test_therapy = therapy.iloc[test_ids,:]
        #therapy_scaler = PowerTransformer()
        #scaled_train_therapy = pd.DataFrame(therapy_scaler.fit_transform(train_therapy), index = train_therapy.index, columns = train_therapy.columns)
        #scaled_val_therapy = pd.DataFrame(therapy_scaler.transform(val_therapy), index = val_therapy.index, columns = val_therapy.columns)
        #scaled_test_therapy = pd.DataFrame(therapy_scaler.transform(test_therapy), index = test_therapy.index, columns = test_therapy.columns)



        train_frequencies = labels_frequencies.iloc[train_ids,:]
        val_frequencies = labels_frequencies.iloc[val_ids,:]
        test_frequencies = labels_frequencies.iloc[test_ids,:]

        train_complications = label_complications.iloc[train_ids,:]
        val_complications = label_complications.iloc[val_ids,:]
        test_complications = label_complications.iloc[test_ids,:]

        ########################
        #save for quality control
        #########################
        scaled_train_diagnostics.to_csv('./results/' + s.mode +'/data/scaled_diagnostic_train' + str(split) + '.csv')
        scaled_val_diagnostics.to_csv('./results/' + s.mode +'/data/scaled_diagnostic_val' + str(split) + '.csv')
        scaled_test_diagnostics.to_csv('./results/' + s.mode +'/data/scaled_diagnostic_test' + str(split) + '.csv')

        therapy.iloc[train_ids,:].to_csv('./results/' + s.mode +'/data/scaled_therapy_train' + str(split) + '.csv')
        therapy.iloc[val_ids,:].to_csv('./results/' + s.mode +'/data/scaled_therapy_val' + str(split) + '.csv')
        therapy.iloc[test_ids,:].to_csv('./results/' + s.mode +'/data/scaled_therapy_test' + str(split) + '.csv')

        scaled_train_labels.to_csv('./results/' + s.mode +'/data/scaled_labels_train' + str(split) + '.csv')
        scaled_val_labels.to_csv('./results/' + s.mode +'/data/scaled_labels_val' + str(split) + '.csv')
        scaled_test_labels.to_csv('./results/' + s.mode +'/data/scaled_labels_test' + str(split) + '.csv')


        ###############################################
        train_ds = DS_Interaction(scaled_train_diagnostics, scaled_train_therapy, scaled_train_labels, train_frequencies,train_complications)
        val_ds = DS_Interaction(scaled_val_diagnostics, scaled_val_therapy, scaled_val_labels, val_frequencies, val_complications)
        test_ds = DS_Interaction(scaled_test_diagnostics, scaled_test_therapy, scaled_test_labels, test_frequencies, test_complications)


        for label_pos in range(labels.shape[1]):

            print(train_ds.labs_features[label_pos], split)


            model = Interaction_Model(train_ds)



            train_test_one_label_overall(model, train_ds, val_ds, test_ds, label_pos=label_pos, batch_size = s.batch_size, n_epochs = s.n_epochs, fold = split, device = s.device)

            test_model(model, test_ds, label_pos=label_pos, fold = split, label_scaler = label_scaler, device = s.device)



            if s.compute_LRP:
                compute_LRP_scores(model, test_ds, label_pos=label_pos, fold=split, device = s.device)


def main():
    for grade in s.grades:
        # downstream modules (training, LRP, test_model) write to results/<s.mode>/,
        # so switch s.mode per grade to keep each grade's outputs in its own folder.
        s.mode = BASE_MODE + '_grade' + str(grade)
        print('==================== grade', grade, '-> results/' + s.mode + ' ====================')
        tc.cuda.empty_cache()
        gc.collect()
        run_grade(grade)

    print("finished")




if __name__ == '__main__':
    print('lets go')
    main()
