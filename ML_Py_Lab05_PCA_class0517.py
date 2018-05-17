# -*- coding: utf-8 -*-

import numpy as np  
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_lfw_people # 사진 이미지 데이터 셋 
from sklearn.neighbors import KNeighborsClassifier # knn 구현 알고리즘
from sklearn.model_selection import train_test_split # 데이터 셋 나누기 클래스 
import mglearn # 파이썬 머신러닝 책 쓰신 분이 만든 패키지 
from sklearn.decomposition import PCA

### 한글
import matplotlib
from matplotlib import font_manager, rc
font_loc = "C:/Windows/Fonts/malgunbd.ttf"
font_name = font_manager.FontProperties(fname=font_loc).get_name()
matplotlib.rc('font', family=font_name)

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
people
image_shape = people.images[0].shape
# target, data, desc...
print(people.DESCR)
print(people.data.shape)   # 3023개.
print(people.target.shape) # target의 배열(3023)
print(people.target_names) # 사진 사람이름 
print(people.images.shape) # 사진 색상 정보 

fig, axes = plt.subplots(3, 5, figsize=(15,8),
           subplot_kw={'xticks':(), 'yticks':()})

a1 = people.target  # 어떤 사람을 가르키는 번호
a2 = people.images  # 사진의 이미지 정보
a3 = axes.ravel()
a3
# ==========================================
for target, image, ax in zip(a1, a2, a3):
    ax.imshow(image)
    ax.set_title(people.target_names[target])



# 데이터의 편중을 없애기 위해 50개만 선택했다.
mask = np.zeros(people.target.shape, dtype=np.bool)
mask

print(people.target.shape)  #전체 데이터 크기
print(mask)
print(np.unique(people.target)) # 나오는 값들 중복 제외하고 뽑기 
print(np.where(people.target==target)[0][:50])


# ==========================================
for target in np.unique(people.target):
    mask[np.where(people.target==target)[0][:50]] = 1

mask
mask[mask==1,].shape

people.data[mask].shape
