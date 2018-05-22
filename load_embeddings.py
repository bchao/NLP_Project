import pickle


root = '/Users/Daniel/Documents/University of Chicago/TTIC 31190/project2.7/'

train_names = ['ROCStories__spring2016 - ROCStories_spring2016',
               'ROCStories_winter2017 - ROCStories_winter2017']
val_name = 'cloze_test_val__spring2016 - cloze_test_ALL_val'
test_name = 'cloze_test_test__spring2016 - cloze_test_ALL_test'

train1 = pickle.load(open(root + 'embeddings_' + train_names[0] + '.pkl', 'rb'), encoding='latin1')
train2 = pickle.load(open(root + 'embeddings_' + train_names[1] + '.pkl', 'rb'), encoding='latin1')
val = pickle.load(open(root + 'embeddings_' + val_name + '.pkl', 'rb'), encoding='latin1')
val_labels = pickle.load(open(root + 'labels_' + val_name + '.pkl', 'rb'), encoding='latin1')
test = pickle.load(open(root + 'embeddings_' + test_name + '.pkl', 'rb'), encoding='latin1')
test_labels = pickle.load(open(root + 'labels_' + test_name + '.pkl', 'rb'), encoding='latin1')
