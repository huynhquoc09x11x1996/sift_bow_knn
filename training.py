from svmutil import *
import cv2
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib
import csv
import time
from datetime import datetime


print("time begin : ",str(datetime.now())[11:])
K=200
full_link=[]
meo = os.listdir("./train/cat/")
for namem in meo:
    full_link.append(["./train/cat/"+namem,1])
cho = os.listdir("./train/dog/")
for namec in cho:
    full_link.append(["./train/dog/"+namec,2])

print("so luong anh = ",len(full_link))

print("extracting...")
hists_ex=[]
hists_ap=[]
for link in full_link:
    img=cv2.imread(link[0],cv2.IMREAD_GRAYSCALE)
    sift = cv2.xfeatures2d.SIFT_create()
    _, des = sift.detectAndCompute(img, None)
    hists_ex.extend(des)
    hists_ap.append([des, link[1]])

print("extracted!")
hists_ap=np.asarray(hists_ap)
hists_ex=np.asarray(hists_ex)

print("train bow..")
kmeans = MiniBatchKMeans(init='k-means++', n_clusters=K, batch_size=20,
                      max_no_improvement=10, verbose=0,init_size=3*K)
kmeans.fit(hists_ex)
# print(kmeans.cluster_centers_)
joblib.dump(kmeans,"kmean.pkl")
print("bow finished!")

print("prepare dataset...")
all_hist=[]
for i,descriptor_i in enumerate(hists_ap):
    histo = np.zeros(K+1)
    for x,des in enumerate(descriptor_i[0]):
        idx = kmeans.predict([des])
        histo[idx] += 1
        histo[-1]=descriptor_i[1]
    all_hist.append(histo)

dataset=np.asarray(all_hist)

print("time trained :" , str(datetime.now())[11:])
print("dataset prepared!")
np.savetxt("dataset.csv", dataset, delimiter=",", fmt='%s')




