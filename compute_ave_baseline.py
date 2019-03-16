import csv
import numpy as np

PREDICTIONPATH = 'prediction/'

test_filelist = 'workingfiles/filelists/test_test'

pred_baseline = csv.reader(open(PREDICTIONPATH+'DCASE_submission_baseline.csv', 'r'))
arr_baseline = {}
for k, r in pred_baseline:
    arr_baseline[k]=r

testfile = open(test_filelist, 'r')
testfilenames = testfile.readlines()
testfile.close()

fidwr = open(PREDICTIONPATH+'DCASE_submission_final.csv', 'wt')
try:
    writer = csv.writer(fidwr)
    for i in range(len(testfilenames)):
        strf = testfilenames[i]
        strf = strf[strf.find('/')+1:-9]
        average_score = np.average(float(arr_baseline[strf]))
        writer.writerow((strf, str(float(average_score))))
finally:
    fidwr.close()
    pred_baseline.close()
