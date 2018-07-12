# -*- coding: utf-8 -*-

# ==================== [1] data read ==================== #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
data16 = pd.read_csv('ambul_data16_2.csv',encoding='cp949')
data16.columns
# 긴급구조시도명이 경기도인 데이터 테이블 생성
data_table = data16[data16['긴급구조시도명'] == '경기도']

# ==================== [2] 파생변수 생성 ==================== #

# (1) 시군명(구이름을 시에 포함한 이름 ex 서울시 노원구 -> 서울시)
data_table['sigun'] = [x.split(" ")[0] for x in data_table['긴급구조시군구명']]

# (2) 시군+읍면동 이름 (ex 서울시 + 노원구 -> 서울시 노원구)
data_table['si_emd'] = data_table['sigun'] + " " + data_table['긴급구조읍면동명']
data_table['si_emdli'] = data_table['si_emd'] + " " + data_table['긴급구조리명']
data_table['emdli'] = data_table['sigun'] + " " + data_table['긴급구조리명']


# ==================== [3] 데이터 필터링 ==================== #
# 전체 사건 데이터 : 535,853건

# 1. 긴급구조읍면동명 없는 값 제외 
data_table = data_table[data_table['긴급구조읍면동명'].notnull()] # 529,727건
data_table = data_table[data_table['긴급구조리명'].notnull()] # 529,689건

# 2. 빈도수가 낮은 여주군, 경기도 제외
data_table = data_table[data_table['sigun'] != '경기도']
data_table = data_table[data_table['sigun'] != '여주군'] # 529,644건

# 3. 현장도착걸린시간 30분 이하 데이터
np.percentile(data16['amb_time'],99) # 99% 데이터가 31분 이하
data_table = data_table[data_table['amb_time']<=30]

# 4. 환자연령이 120세 이하
data_table = data_table[data_table['환자연령']<=120] # 최종 442,864건

# ==================== [4] 데이터 필터링 ==================== #


# ==================== [4] 지역 테이블 생성(동리) ==================== #
# ===  1. Numeric value (시군+읍면동+동리 기준으로 groupby) === #
# 1) 지역별 사건 빈도 수
emd_df = pd.DataFrame(data_table.groupby('si_emdli').size(),
                      columns=['freq_119']).reset_index()
# 2) 지역별 사건 평균 시간 !!! (평균이 적당한지 추후 확인 필요)
emd_df['avg_time'] = data_table.groupby('si_emdli')['amb_time'].mean().values
# 3) 지역별 평균 사건 거리 및 거리의 총 합 !!!
emd_df['avg_dist'] = data_table.groupby('si_emdli')['dist_round'].mean().values
emd_df['sum_dist'] = data_table.groupby('si_emdli')['dist_round'].sum().values
# 4) 지역별 평균 환자연령
emd_df['avg_age'] = data_table.groupby('si_emdli')['환자연령'].mean().values

# ===  2. categorycal value === #
# 1) 지역별 환자 성별 사건 빈도 (시군+읍면동 기준으로 pivot table)
piv_df1 = pd.pivot_table(data_table,index=['si_emdli'],columns=['환자성별구분명'],
               values=['시군명'],aggfunc=np.size).reset_index()
piv_df1.columns =['si_emdli','male','noen_gender','female']
# 2) 지역별 구급발생장소유형 빈도
piv_df2 = pd.pivot_table(data_table,index=['si_emdli'],columns=['구급발생장소유형'],
               values=['시군명'],aggfunc=np.size).reset_index()
piv_df2.columns =['si_emdli','가정', '강바다', '고속도로', '공공장소', '공사장', '공장', '기타장소', '논.밭', '도로', '병원', '사무실', '산', '숙박시설', '스키장', '식당', '유흥장소', '일반도로', '주택가', '지하철', '체육시설', '학교']
# 3) 지역별 환자 성별 사건 빈도
piv_df3 = pd.pivot_table(data_table,index=['si_emdli'],columns=['구급사고종별상위명'],
               values=['환자연령'],aggfunc=np.size).reset_index()
piv_df3.columns =['si_emdli','교통사고', '기타사고', '사고부상', '질병']
# 4) 지역별 환자 성별 사건 빈도
piv_df4 = pd.pivot_table(data_table,index=['si_emdli'],columns=['관할구분명'],
               values=['시군명'],aggfunc=np.size).reset_index()
piv_df4.columns =['si_emdli','센터', '센터외']

# =  numeric & categorical table merge = #
merged_df = pd.merge(emd_df,piv_df1,on='si_emdli')
merged_df = pd.merge(merged_df,piv_df2,on='si_emdli',how='left')
merged_df = pd.merge(merged_df,piv_df3,on='si_emdli',how='left')
merged_df = pd.merge(merged_df,piv_df4,on='si_emdli',how='left')    

# ==================== [5] 지역 테이블 파생 ==================== #
# 1. 시군명 변수
merged_df['sigun'] =  [x.split(" ")[0] for x in merged_df['si_emdli']]

# 2. 읍면동 변수
merged_df['emd'] = [x.split(" ")[1] for x in merged_df['si_emdli']]

# 3. 동리 변수
merged_df['li'] = [x.split(" ")[2] for x in merged_df['si_emdli']]
# 4. 시군명 + 읍면동, 시군명 + 동리
merged_df['sigun_emd'] = merged_df['sigun'] + " " + merged_df['emd']
merged_df['sigun_li'] = merged_df['sigun'] + " " + merged_df['li']

merged_df.columns
# clustering data
merged_df = merged_df[['sigun','emd','li','sigun_emd','sigun_li', 'freq_119', 'avg_time', 'avg_dist', 'sum_dist', 
                       'avg_age','male', 'noen_gender', 'female', '가정', '강바다', '고속도로', '공공장소', '공사장',
                       '공장', '기타장소', '논.밭', '도로', '병원', '사무실', '산', '숙박시설', '스키장', '식당',
                       '유흥장소', '일반도로', '주택가', '지하철', '체육시설', '학교', '교통사고', '기타사고', '사고부상',
                       '질병', '센터', '센터외']]


# ==================== [6] 지역 테이블 인구 ==================== #
# pop data load
pop_16 = pd.read_csv(r'C:\Users\KEJ\Desktop\ambulance\emd_raw_pop.csv',encoding='cp949')
groub_pop = pop_16.groupby('sigun_li').sum()
pop16_df = pd.DataFrame(groub_pop).reset_index()

merged_pop_df = pd.merge(merged_df,pop16_df,on='sigun_li')
# merged_pop_df.to_csv(r'C:\Users\KEJ\Desktop\ambulance\ambul_df_emdli.csv',index = False)

# ==================== [7] 지역 테이블 생성(읍면동) 5번과정을 다시함(평균때문에 다시 계산필요) ==================== #

merged_pop_df.columns
name_valList = ['sigun','sigun_emd']
sum_valList = ['freq_119',  'sum_dist',
       'male', 'noen_gender', 'female', '가정', '강바다', '고속도로', '공공장소', '공사장',
       '공장', '기타장소', '논.밭', '도로', '병원', '사무실', '산', '숙박시설', '스키장', 
       '식당',
       '유흥장소', '일반도로', '주택가', '지하철', '체육시설', '학교', '교통사고', '기타사고', 
       '사고부상',
       '질병', '센터', '센터외','pop']
mean_valList = ['avg_time', 'avg_dist', 'avg_age']

emd_df2 = pd.DataFrame(merged_pop_df.groupby('sigun_emd')[sum_valList].sum(),
       columns = sum_valList)
emd_df2[mean_valList] = merged_pop_df.groupby('sigun_emd')[mean_valList].mean()
emd_df2 = emd_df2.reset_index()

#  시군명 변수 추가
emd_df2['sigun'] =  [x.split(" ")[0] for x in emd_df2['sigun_emd']]
# emd_df2.to_csv(r'C:\Users\KEJ\Desktop\ambulance\ambul_df_emd.csv',index = False)


# ==================== [6] 지역 테이블(인구 읍면동) 빈도 -> 비율 변환 ==================== #
# nan 값 0으로 처리
emd_df2 = emd_df2.fillna(0)

# rate 변환 함수 정의
def makeRate(input_val):
    output_val = pd.DataFrame()
    input_col = input_val
    col = input_col.columns 
    
    for i in range(len(col)):
        output_val[col[i]] = (input_col[col[i]] / np.sum(input_col,axis=1))
    
    return output_val

emd_df2.columns
['sigun_emd', 'freq_119', 'sum_dist', 'male', 'noen_gender', 'female',
       '가정', '강바다', '고속도로', '공공장소', '공사장', '공장', '기타장소', '논.밭', '도로', '병원',
       '사무실', '산', '숙박시설', '스키장', '식당', '유흥장소', '일반도로', '주택가', '지하철', '체육시설',
       '학교', '교통사고', '기타사고', '사고부상', '질병', '센터', '센터외', 'pop', 'avg_time',
       'avg_dist', 'avg_age', 'sigun']
# 변환하지 않는 변수
emd_df3 = emd_df2[['sigun_emd', 'freq_119', 'sum_dist','pop', 'avg_time',
                  'avg_dist', 'avg_age', 'sigun']]
# 변환 변수 구분
gender = ['male','female']

case_place = ['가정', '강바다', '고속도로', '공공장소', '공사장', '공장', '기타장소', '논.밭', '도로', '병원',
       '사무실', '산', '숙박시설', '스키장', '식당', '유흥장소', '일반도로', '주택가', '지하철', '체육시설',
       '학교']
case_category = ['교통사고', '기타사고', '사고부상', '질병']
center = ['센터', '센터외']

# 변환
emd_df3[gender] = makeRate(emd_df2[gender])
emd_df3[case_place] = makeRate(emd_df2[case_place])
emd_df3[case_category] = makeRate(emd_df2[case_category])
emd_df3[center] = makeRate(emd_df2[center])













