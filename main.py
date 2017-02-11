import loader
import numpy
import math
import rf
import time
# train data
dix, dsize, _ = loader.readImage("D:\\TCD\\leftImg8bit\\train\\aachen\\aachen_000000_000019_leftImg8bit.png")
# training labels
cix, csize, _ = loader.readLabels("D:\\TCD\\gtFine\\train\\aachen\\aachen_000000_000019_gtFine_labelIds.png")
# test data
tix, tsize, _ = loader.readImage("D:\\TCD\\leftImg8bit\\train\\bochum\\bochum_000000_000313_leftImg8bit.png")
# testing labels
uix, usize, _ = loader.readLabels("D:\\TCD\\gtFine\\train\\bochum\\bochum_000000_000313_gtFine_labelIds.png")
if dsize != csize:
    print 'Image size mismatch. Abort.'
    exit(1)
# parameters
patch_size = 5

step = int(math.floor(patch_size/2.0));
all_data = []
all_label = []

t = time.time()
myRF = rf.init()
#for i in range(step, dsize[0] - step):
#    for j in range(step, dsize[1] - step):
for i in numpy.random.randint(step, dsize[0] - step, size=200):
    for j in numpy.random.randint(step, dsize[1] - step, size=200):
        data_entry = []
        for k in range(3):
            data = [ [ dix[m, n][k] for m in range(i-step, i+step+1) ] for n in range(j-step, j+step+1) ]
            flat_data = [item for sublist in data for item in sublist]
            data_entry.extend(flat_data)
            # data = dix[i-step:i+step, j-step:j+step].flatten()
            # print r[0, 0]
            # data = r[i - step:i + step, j - step:j + step].flatten()
        all_data.append(data_entry)
        all_label.append(cix[i, j])
elapsed = time.time() - t
print "Data preparation completed after {:.5} seconds".format(elapsed)

elapsed = time.time() - t
t = time.time()
myRF = rf.init()
myRF.fit(all_data, all_label)
elapsed = time.time() - t
print "Fit completed after {:.5} seconds".format(elapsed)

test_data = []
test_label = []
for i in numpy.random.randint(step, dsize[0] - step, size=100):
    for j in numpy.random.randint(step, dsize[1] - step, size=100):
        data_entry = []
        for k in range(3):
            data = [[tix[m, n][k] for m in range(i - step, i + step + 1)] for n in range(j - step, j + step + 1)]
            flat_data = [item for sublist in data for item in sublist]
            data_entry.extend(flat_data)
            # data = dix[i-step:i+step, j-step:j+step].flatten()
            # print r[0, 0]
            # data = r[i - step:i + step, j - step:j + step].flatten()
        test_data.append(data_entry)
        test_label.append(uix[i, j])

t = time.time()
Y = myRF.predict(test_data)
correct_list = [i for i in range(len(Y)) if test_label[i] == Y[i]]
elapsed = time.time() - t
print "Predict completed after {:.5} seconds".format(elapsed)
print "Predict correctly {} out of {}".format(len(correct_list), len(test_label))

print all_label
print test_label

# labels = numpy.random.randint(1, 5, size= (1024, 1024))
# loader.printLabels(labels)