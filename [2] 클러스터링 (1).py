# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 14:29:44 2017

@author: KEJ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn import preprocessing
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# ==================== [1] 함수 정의 ==================== #
def elbow(X):
    sse = []
    for i in range(1,11):
        km = KMeans(n_clusters=i, init ='k-means++',random_state=0)
        km.fit(X)
        sse.append(km.inertia_)
        
    plt.plot(range(1,11),sse,marker='o')
    plt.xlabel('cluster size')
    plt.ylabel('SSE')
    plt.show()

from sklearn.metrics import silhouette_samples
from matplotlib import cm

def plotSilhouette(X,y_km):
    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    Silhouette_vals =silhouette_samples(X, y_km, metric = 'euclidean')
    y_ax_lower, y_ax_upper =0,0
    yticks =[]
    
    for i,c in enumerate(cluster_labels):
        c_Silhouette_vals = Silhouette_vals[y_km == c]
        c_Silhouette_vals.sort()
        y_ax_upper += len(c_Silhouette_vals)
        color = cm.jet(i/n_clusters)
        
        plt.barh(range(y_ax_lower, y_ax_upper), c_Silhouette_vals, height=1.0,
                 edgecolor='none', color=color)
        yticks.append((y_ax_lower + y_ax_upper)/2)
        y_ax_lower += len(c_Silhouette_vals)
        
    Silhouette_avg = np.mean(Silhouette_vals)
    plt.axvline(Silhouette_avg,color='red',linestyle='--')
    plt.yticks(yticks,cluster_labels+1)
    plt.ylabel('cluster')
    plt.xlabel('실루엣계수')
    plt.show()
    
# ==================== [1] data read ==================== #
read_dir = r'C:\Users\KEJ\Desktop\ambulance\생성 데이터 테이블'
df_emd = pd.read_csv(read_dir + '\\ambul_df_emd_rate.csv',encoding = 'cp949')
df_emd.columns

# 클러스터링 input data
cls_input_columns = ['freq_119', 'sum_dist', 'pop', 'avg_time', 'avg_dist',
       'avg_age', 'male', 'female', '가정', '강바다', '고속도로', '공공장소',
       '공사장', '공장', '기타장소', '논.밭', '도로', '병원', '사무실', '산', '숙박시설', '스키장', '식당',
       '유흥장소', '일반도로', '주택가', '지하철', '체육시설', '학교', '교통사고', '기타사고', '사고부상',
       '질병', '센터', '센터외']

#  클러스터링 input 변수 그룹화 (!!! 조합 생각해 볼 것)
# (1) p 사고발생장소 유형의 그룹화
# 그룹정의
p_house = ['가정','주택가'] 
p_nature = ['강바다', '산','논.밭']

p_way = ['일반도로','도로','고속도로','지하철']
p_factory = ['공사장', '공장']
p_facility = ['공공장소','병원', '사무실','숙박시설', 
              '스키장', '식당','유흥장소','체육시설', '학교']
p_etc =['기타장소']

p_group_list = [p_house,p_nature,p_way,p_factory,p_facility]
p_group_list_name = ['p_house','p_nature','p_way','p_factory','p_facility']

# 정의된 그룹 적용한 새로운 테이블 생성
df_emd_group = df_emd[['freq_119', 'sum_dist', 'pop', 'avg_time', 'avg_dist',
       'avg_age', 'male', 'female','교통사고', '기타사고', '사고부상',
       '질병', '센터', '센터외']]

for i in range(len(p_group_list)):
    df_emd_group[p_group_list_name[i]] = np.sum(df_emd[p_group_list[i]],axis=1)

## 결과 csv파일 저장
# output_dir = r'C:\Users\KEJ\Desktop\ambulance'
# df_emd_group.to_csv(output_dir + '\\group_df_emd.csv',index = False)


cls_df = df_emd_group.copy()

# ==================== [2] 전처리 ==================== #
# nan 값 0처리
cls_df.fillna(0,inplace=True)

# (case1)전체 변수 표준화
X_zscore = stats.zscore(cls_df)
X_zscore = pd.DataFrame(X_zscore)

"""
# (case2) 비율 값 제외한 나머지 표준화 (!!! 비율도 표준화해야하는지 모르겠음...추가확인필요)
cls_df2 = cls_df.copy()
s_columns = ['freq_119', 'sum_dist', 'pop', 'avg_time', 'avg_dist','avg_age']
cls_df2[s_columns] = stats.zscore(cls_df2[s_columns])
"""

# ==================== [3] 클러스터링 개수 ==================== #
# elobow
elbow(X_zscore) 
# elbow(cls_df2) # 전체 표준화와 비율제외 표준화의 엘보우 값이 다름

# 실루엣 계수 및 그래프
sil_list =[]
for i in range(2,15):
    cls = KMeans(n_clusters=i,init ='k-means++',random_state=0)
    cls.fit(X_zscore)   
    cls_label = pd.Series(cls.labels_,index=X_zscore.index)
    # print("cluster :",i," silhouette_score :",silhouette_score(X_zscore,cls_label))
    sil_list.append(silhouette_score(X_zscore,cls_label))

plt.plot(range(2,15),sil_list,marker='o')
plt.xlabel('cluster size')
plt.ylabel('silhouette_score')


# ==================== [4] 클러스터링 ==================== #
# 군집 수 7개
cls = KMeans(n_clusters=7,init ='k-means++',random_state=0)
cls.fit(X_zscore)
cls_label = pd.Series(cls.labels_,index=X_zscore.index)


# ==================== [5] 클러스터링 확인 ==================== #
# 그래프 확인 (x축 시간, y축 사건빈도, 색 = 군집)
plt.scatter(X_zscore[[3]],X_zscore[[0]],c=cls_label)

# 군집별 지역 수 확인
for i in range(len(cls.cluster_centers_)):
    print("cluster :",i," clust_num :",len(X_zscore[cls_label==i]))

# 군집별 통계량 확인
# cls_14df_over_table['clust'] = str(cls_label) # 클래스 인덱스가 바뀌었는지는 추후 확인필요
check_table_cls = df_emd.copy()
check_table_cls['clust'] = [str(x) for x in cls_label.values] # 라벨링

plt.scatter(check_table_cls[check_table_cls['clust']==0][[3]],
            check_table_cls[check_table_cls['clust']==0][[0]],
            c=cls_label)

# 결과 테이블 csv파일로 저장
output_dir = r'C:\Users\KEJ\Desktop\ambulance'
check_table_cls.to_csv(output_dir + '\\five_cls_totalval.csv',index = False)




