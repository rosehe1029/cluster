import pandas as pd

cons_no=pd.read_csv(r"new_data\7类\6.csv")
#print(cons_no.head())
cons_no=list(cons_no.iloc[:,0])
#print(cons_no)
data=pd.read_csv(r"new_data\7898tmp_pjy_e_mp_power_result_202304131050.csv")

new=[]
for  index ,row in data.iterrows():
    print(index ,row['cons_no'])
    if row['cons_no'] in cons_no:
          new.append(row)
    #break

pd.DataFrame(new).to_csv(r"new_data\7类24列\6.csv",index=False)

