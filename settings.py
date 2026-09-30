location = 'M'
mode = 'mock'
model_type = 'interaction'

# Adverse-event grades to run in one go (Main.py loops over these).
# Grade does not change the regression targets (nadir lab values); it only sets how
# extreme a nadir must be to count as a rare event, which drives resampling. A lower
# quantile => rarer events => stronger oversampling. Grade 3 is rarer than grade 1.
grades = [1, 3]
grade_event_quantile = {1: 0.30, 3: 0.10}

if location == 'M':
    n_product = 1000
    hidden_factor = 10

    n_splits = 5

    batch_size=32
    n_epochs = 50
    epochs_until_restart = 10
    starting_lr = 1e-4
    

    input_dropout = 0.0
    intermediate_dropout = 0.1

    device = 'cpu'

    compute_LRP = True
    only_select_given_therapies = True
    fit_all_labels = True
    resampling = True
    regression_loss = True

    
     