#  -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans

# 主程序
def main():
    # 读取数据文件
    readPath = r"new_data\data1.csv"  # 数据文件的地址和文件名
    dfFile = pd.read_csv(readPath, header=0)  # 首行为标题行
    dfFile = dfFile.fillna(method='ffill')
    dfFile=dfFile.fillna(method='bfill')
    print(dfFile.head())

    # 数据准备
    #z_scaler = lambda x:(x-np.mean(x))/np.std(x)  # 定义数据标准化函数
    dfScaler = dfFile[['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100','101','102','103','104','105','106','107','108','109','110','111','112','113','114','115','116','117','118','119','120','121','122','123','124','125','126','127','128','129','130','131','132','133','134','135','136','137','138','139','140','141','142','143','144','145','146','147','148','149','150','151','152','153','154','155','156','157','158','159','160','161','162','163','164','165','166','167','168','169','170','171','172','173','174','175','176','177','178','179','180','181','182','183','184','185','186','187','188','189','190','191']]#.apply(z_scaler)  # 数据归一化
    dfData = pd.concat([dfFile[['cons_no']], dfScaler], axis=1)  # 列级别合并
    df = dfData.loc[:,['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100','101','102','103','104','105','106','107','108','109','110','111','112','113','114','115','116','117','118','119','120','121','122','123','124','125','126','127','128','129','130','131','132','133','134','135','136','137','138','139','140','141','142','143','144','145','146','147','148','149','150','151','152','153','154','155','156','157','158','159','160','161','162','163','164','165','166','167','168','169','170','171','172','173','174','175','176','177','178','179','180','181','182','183','184','185','186','187','188','189','190','191']]  # 基于全部 10个特征聚类分析
    # df = dfData.loc[:,['x1','x2','x7','x8','x9','x10']]  # 降维后选取 6个特征聚类分析
    X = np.array(df)  # 准备 sklearn.cluster.KMeans 模型数据
    print("Shape of cluster data:", X.shape)

    # KMeans 聚类分析(sklearn.cluster.KMeans)
    nCluster = 7 #7#6#5#4#3#2
    kmCluster = KMeans(n_clusters=nCluster).fit(X)  # 建立模型并进行聚类，设定 K=2
    print("Cluster centers:\n", kmCluster.cluster_centers_)  # 返回每个聚类中心的坐标
    print(type(kmCluster.cluster_centers_))
    #pd.DataFrame(kmCluster.cluster_centers_).to_csv(r"聚类中心.csv")
    #print("Cluster results:\n", kmCluster.labels_)  # 返回样本集的分类结果
    
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import silhouette_samples
 

    y=kmCluster.predict(X)
    print('y',y)
    # 查看轮廓系数均值
    #print(silhouette_score(X,kmCluster.labels_))

    ''' 
    # 整理聚类结果
    listName = dfData['cons_no'].tolist()  # 将 dfData 的首列 '地区' 转换为 listName
    dictCluster = dict(zip(listName,kmCluster.labels_))  # 将 listName 与聚类结果关联，组成字典
    listCluster = [[] for k in range(nCluster)]
    for v in range(0, len(dictCluster)):
        k = list(dictCluster.values())[v]  # 第v个城市的分类是 k
        listCluster[k].append(list(dictCluster.keys())[v])  # 将第v个城市添加到 第k类
    print("\n聚类分析结果(分为{}类):".format(nCluster))  # 返回样本集的分类结果
    for k in range(nCluster):
        print("第 {} 类：{}".format(k, listCluster[k]))  # 显示第 k 类的结果
    
    #存储
    with open(r'new_data\7类\00.csv', 'w') as f:
        [f.write('{0},{1}\n'.format(key, value)) for key, value in dictCluster.items()]
    
    pd.DataFrame(listCluster).T.to_csv(r"new_data\7类\01.csv",index=False)
    '''
    return

if __name__ == '__main__':
    main()


