from __future__ import print_function

# these lines just make it work in my PyCharm console
import sys
sys.path.append("/Users/Daniel/Documents/University of Chicago/TTIC 31190/project/skip-thoughts")


# need to update file paths in skipthoughts.py before running
import skipthoughts
import csv
import pickle
import time


model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)

root = '/Users/Daniel/Documents/University of Chicago/TTIC 31190/project/'

# read CSV files and save sentence embeddings
# first do the training data
# output format:
#   [[s1vec, s2vec, s3vec, s4vec, s5vec], ..., [s1vec, s2vec, s3vec, s4vec, s5vec]]
# takes 1-2 hours per file on my computer; output is 12+ GB per file
names = ['ROCStories__spring2016 - ROCStories_spring2016',
         'ROCStories_winter2017 - ROCStories_winter2017']
for name in names:
    print("\nstarting " + name + " - " + time.asctime())
    fname = root + name + '.csv'
    with open(fname) as f:
        # read contents of CSV into dictionaries
        reader = csv.reader(f)
        data = [r for r in reader]
        data.pop(0)  # remove first line (headers)

        X = []
        for line in data:
            X.extend(line[2:7])  # add sentences 1 through 5 (skip storyid and storytitle)

        # get embeddings for each sentence, then split into groups of 5
        out = encoder.encode(X, verbose=False)
        out = [out[i:i+5] for i in range(0, len(out), 5)]

        f.close()

    # write to file
    print("writing output - " + time.asctime())
    with open(root + 'embeddings_' + name + '.pkl', 'wb') as out_file:
        pickle.dump(out, out_file)

print("finished - ", time.asctime())

# now do the validation and testing data (different format)
# two outputs:
#   out: [[s1vec, s2vec, s3vec, s4vec, s5.1vec, s5.2vec], ..., [s1vec, s2vec, s3vec, s4vec, s5.1vec, s5.2vec]]
#   labels: [answer1, answer2, ..., answerN]
# takes 2-3 minutes per file on my computer; output is about 600 MB per file
names = ['cloze_test_val__spring2016 - cloze_test_ALL_val',
         'cloze_test_test__spring2016 - cloze_test_ALL_test']
for name in names:
    print("\nstarting " + name + " - " + time.asctime())
    fname = root + name + '.csv'
    with open(fname) as f:
        # read contents of CSV into dictionaries
        reader = csv.reader(f)
        data = [r for r in reader]
        data.pop(0)  # remove first line (headers)

        X = []
        labels = []
        for line in data:
            X.extend(line[1:7])  # add sentences 1 through 4 and the two options for sentence 5
            labels.append(line[7])

        # get embeddings for each sentence, then split into groups of 6
        out = encoder.encode(X, verbose=False)
        out = [out[i:i + 6] for i in range(0, len(out), 6)]

        f.close()

    # write to file
    print("writing output - " + time.asctime())
    with open(root + name + '.pkl', 'wb') as out_file:
        pickle.dump(out, out_file)
    with open(root + 'labels_' + name + '.pkl', 'wb') as label_file:
        pickle.dump(labels, label_file)

print("finished - ", time.asctime())
