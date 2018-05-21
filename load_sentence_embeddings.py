import pickle


root = '/Users/Daniel/Documents/University of Chicago/TTIC 31190/project/'
train_names = ['ROCStories__spring2016 - ROCStories_spring2016',
               'ROCStories__winter2017 - ROCStories_winter2017']
val_name = 'cloze_test_val__spring2016 - cloze_test_ALL_val'
test_name = 'cloze_test_test__spring2016 - cloze_test_ALL_test'

train1 = pickle.load(open(root + train_names[0] + '.pkl', 'rb'))
train2 = pickle.load(open(root + train_names[1] + '.pkl', 'rb'))
val = pickle.load(open(root + val_name + '.pkl', 'rb'))
val_labels = pickle.load(open(root + 'labels_' + val_name + '.pkl', 'rb'))
test = pickle.load(open(root + test_name + '.pkl', 'rb'))
test_labels = pickle.load(open(root + 'labels_' + test_name + '.pkl', 'rb'))
