import pickle
from sklearn.neighbors import NearestNeighbors
import numpy as np

with open("training_x.dat", 'rb') as pickleFile:
     x_tr = pickle.load(pickleFile)
with open("training_y.dat", 'rb') as pickleFile:
     y_tr = pickle.load(pickleFile)
with open("validation_x.dat", 'rb') as pickleFile:
     x_val = pickle.load(pickleFile)

print(x_tr[])
clf = NearestNeighbors(n_neighbors=1,algorithm='kd_tree')
#clf.fit(x_tr, y_tr)
#y_pred = clf.predict(x_val)