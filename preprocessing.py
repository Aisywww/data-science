import numpy as np 
from sklearn import preprocessing

raw_csv_data = np.loadtxt('C:/Users/Lenovo/Downloads/Audiobooks_data.csv',delimiter = ',')

unsclaed_inputs_all = raw_csv_data[:,1:-1]
targets_all = raw_csv_data[:,-1]
print(targets_all.shape[0])
print(unsclaed_inputs_all)

num_one_targets = int(np.sum(targets_all))
zero_targets_counter = 0
indices_to_remove = []

for i in range(targets_all.shape[0]):
    if targets_all[i] ==0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)
            
unsclaed_inputs_equal_priors = np.delete(unsclaed_inputs_all,indices_to_remove,axis = 0)
targets_equal_priors = np.delete(targets_all,indices_to_remove,axis = 0)

scaled_inputs = preprocessing.scale(unsclaed_inputs_equal_priors)

print(scaled_inputs.shape[0])
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]

sampel_count = shuffled_inputs.shape[0]

train_sample_count = int(0.8*sampel_count)
validation = int(0.1*sampel_count)
test_sample_count = sampel_count - train_sample_count - validation

train_inputs = shuffled_inputs[:train_sample_count]
train_targets = shuffled_targets[:train_sample_count]

validationin = shuffled_inputs[train_sample_count:train_sample_count+validation]
validationtar = shuffled_targets[train_sample_count:train_sample_count+validation]

test_in = shuffled_inputs[train_sample_count+validation:]
test_tar = shuffled_targets[train_sample_count+validation:]

print(np.sum(train_targets),train_sample_count,np.sum(train_targets)/train_sample_count)
print(np.sum(validationtar),validation,np.sum(validationtar)/validation)
print(np.sum(test_tar),test_sample_count,np.sum(test_tar)/test_sample_count)

np.savez('Audiobooks_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Audiobooks_data_validation', inputs=validationin, targets=validationtar)
np.savez('Audiobooks_data_test', inputs=test_in, targets=test_tar)

