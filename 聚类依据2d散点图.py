#  -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号


# 使用TSNE进行数据降维并展示聚类结果
from sklearn.manifold import TSNE
tsne = TSNE()
data=pd.read_csv(r"new_data\7类\7类.csv")
data=data.set_index(["cons_no"])
#print(data.head())
X=data.iloc[:,1:]
tsne.fit_transform(X)  # 进行数据降维
# tsne.embedding_可以获得降维后的数据
#print('tsne.embedding_: \n', tsne.embedding_)
tsne = pd.DataFrame(tsne.embedding_, index=data.index)  # 转换数据格式
#print('tsne: \n', tsne)

TYPE=data['type']
output_data=pd.concat([TYPE,tsne],axis=1)  
print(output_data.head()) 

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
nCluster=7
    
# 不同类别用不同颜色和样式绘图
color_style = ["orange",'green','blue','gray','purple','darkcyan',"thistle" ]##   #  
for i in range(nCluster):
    d = tsne[output_data[u'type'] == i]
    # dataframe格式的数据经过切片之后可以通过d[i]来得到第i列数据
    plt.scatter( x=d[0],y=d[1], c=color_style[i], label='第' + str(i)+"类")
plt.legend()
#
plt.show()



