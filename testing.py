import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import csv
import pandas as pd


df=pd.read_csv('dataset.csv', sep=',',header=None)

kmean=joblib.load("kmean.pkl")
dataset=df.values.tolist()

new_img_link="./test/cat.308.jpg"
new_img=cv2.imread(new_img_link,0)
sift = cv2.xfeatures2d.SIFT_create()
_, des = sift.detectAndCompute(new_img, None)

newhist=[]
histo=np.zeros(kmean.n_clusters)
for x, des in enumerate(des):
    idx = kmean.predict([des])
    histo[idx] += 1
newhist.append(histo)


print("predicting...")
new_vec_img=newhist[0].tolist()
dists=[]
for vec_in_dataset in dataset:
    dists.append(np.linalg.norm(np.asarray(new_vec_img)-np.asarray(vec_in_dataset[:-1])))


dict_classes={
    1.0:"Con Meo",
    2.0:"Con Cho"
}
print("predicted!")
new_img=cv2.cvtColor(new_img,cv2.COLOR_GRAY2RGB)
cv2.putText(new_img,dict_classes[dataset[np.argsort(dists)[0]][-1]], (5,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),3)
cv2.imshow("Ket qua du doan", new_img)
cv2.waitKey()
cv2.destroyAllWindows()
# print("Ket qua du doan day la ",dict_classes[dataset[np.argsort(dists)[0]][-1]])
