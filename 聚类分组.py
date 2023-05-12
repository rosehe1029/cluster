import pandas as pd

fenlei=pd.read_csv(r"new_data\7类\01.csv")
#print(fenlei.head())
fenlei0=fenlei['6'].values
#print(fenlei0.values)

data=pd.read_csv("new_data\data1.csv")


new=[]
for index,row in data.iterrows():
    #print(index,type(row),row[0],row[1]) 
    #print(row[0])
    if str(row[0])  in  fenlei0:
        print(row[0])
        new.append(row)


pd.DataFrame(new).to_csv(r"new_data\7类\6.csv",index=False)