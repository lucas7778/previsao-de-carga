# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 22:34:11 2020

@author: lucas
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler,PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout,LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam,RMSprop,SGD
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score,TimeSeriesSplit
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasRegressor

df=pd.read_csv('selected_data_ISONE.csv')

# Transformar a coluna de dias das semanas em outras sete com valores binários  
df1=pd.read_csv('selected_data_ISONE.csv')
df2=pd.read_csv('selected_data_ISONE.csv')
df3=pd.read_csv('selected_data_ISONE.csv')
df4=pd.read_csv('selected_data_ISONE.csv')
df5=pd.read_csv('selected_data_ISONE.csv')
df6=pd.read_csv('selected_data_ISONE.csv')
df7=pd.read_csv('selected_data_ISONE.csv')

df1['dom']=df1['weekday']
df1[df1['dom']!= 1]=0
df1[df1['dom']==1]=1
df2['seg']=df2['weekday']
df2[df2['seg']!= 1]=0
df2[df2['seg']==1]=1
df3['ter']=df3['weekday']
df3[df3['ter']!= 3]=0
df3[df3['ter']==3]=1
df4['qua']=df4['weekday']
df4[df4['qua']!= 4]=0
df4[df4['qua']==4]=1
df5['qui']=df5['weekday']
df5[df5['qui']!= 5]=0
df5[df5['qui']==5]=1
df6['sex']=df6['weekday']
df6[df6['sex']!= 6]=0
df6[df6['sex']==6]=1
df7['sab']=df7['weekday']
df7 [df7['sab']!= 7]=0
df7[df7['sab']==7]=1
df1.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df2.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df3.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df4.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df5.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df6.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df7.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df.drop(['date','weekday','year','month','day'],axis=1,inplace=True)
dataset = df.join([df1,df2,df3,df4,df5,df6,df7])


# Transformar a coluna de dias das horas em outras vinte e quatro com valores binários  

df1=pd.read_csv('selected_data_ISONE.csv')
df2=pd.read_csv('selected_data_ISONE.csv')
df3=pd.read_csv('selected_data_ISONE.csv')
df4=pd.read_csv('selected_data_ISONE.csv')
df5=pd.read_csv('selected_data_ISONE.csv')
df6=pd.read_csv('selected_data_ISONE.csv')
df7=pd.read_csv('selected_data_ISONE.csv')
df8=pd.read_csv('selected_data_ISONE.csv')
df9=pd.read_csv('selected_data_ISONE.csv')
df10=pd.read_csv('selected_data_ISONE.csv')
df11=pd.read_csv('selected_data_ISONE.csv')
df12=pd.read_csv('selected_data_ISONE.csv')
df13=pd.read_csv('selected_data_ISONE.csv')
df14=pd.read_csv('selected_data_ISONE.csv')
df15=pd.read_csv('selected_data_ISONE.csv')
df16=pd.read_csv('selected_data_ISONE.csv')
df17=pd.read_csv('selected_data_ISONE.csv')
df18=pd.read_csv('selected_data_ISONE.csv')
df19=pd.read_csv('selected_data_ISONE.csv')
df20=pd.read_csv('selected_data_ISONE.csv')
df21=pd.read_csv('selected_data_ISONE.csv')
df22=pd.read_csv('selected_data_ISONE.csv')
df23=pd.read_csv('selected_data_ISONE.csv')
df24=pd.read_csv('selected_data_ISONE.csv')

df1[1]=df1['hour']
df1[df1[1]!= 1]=0
df1[df1[1]==1]=1
df2[2]=df2['hour']
df2[df2[2]!= 2]=0
df2[df2[2]==2]=1
df3[3]=df3['hour']
df3[df3[3]!= 3]=0
df3[df3[3]==3]=1
df4[4]=df4['hour']
df4[df4[4]!= 4]=0
df4[df4[4]==4]=1
df5[5]=df5['hour']
df5[df5[5]!= 5]=0
df5[df5[5]==5]=1
df6[6]=df6['hour']
df6[df6[6]!= 6]=0
df6[df6[6]==6]=1
df7[7]=df7['hour']
df7 [df7[7]!= 7]=0
df7[df7[7]==7]=1
df8[8]=df8['hour']
df8[df8[8]!= 8]=0
df8[df8[8]==8]=1
df9[9]=df9['hour']
df9[df9[9]!= 9]=0
df9[df9[9]==9]=1
df10[10]=df10['hour']
df10[df10[10]!= 10]=0
df10[df10[10]==10]=1
df11[11]=df11['hour']
df11[df11[11]!= 11]=0
df11[df11[11]==5]=1
df12[12]=df12['hour']
df12[df12[12]!= 12]=0
df12[df12[12]==12]=1
df13[13]=df13['hour']
df13 [df13[13]!= 13]=0
df13[df13[13]==13]=1
df14[14]=df14['hour']
df14[df14[14]!= 14]=0
df14[df14[14]==14]=1
df15[15]=df15['hour']
df15[df15[15]!= 15]=0
df15[df15[15]==15]=1
df16[16]=df16['hour']
df16 [df16[16]!= 16]=0
df16[df16[16]==16]=1
df17[17]=df17['hour']
df17[df17[17]!= 17]=0
df17[df17[17]==17]=1
df18[18]=df18['hour']
df18[df18[18]!= 18]=0
df18[df18[18]==18]=1
df19[19]=df19['hour']
df19[df19[19]!= 19]=0
df19[df19[19]==19]=1
df20[20]=df20['hour']
df20[df20[20]!= 20]=0
df20[df20[20]==20]=1
df21[21]=df21['hour']
df21[df21[21]!= 21]=0
df21[df21[21]==21]=1
df22[22]=df22['hour']
df22[df22[22]!= 22]=0
df22[df22[22]==22]=1
df23[23]=df23['hour']
df23[df23[23]!= 23]=0
df23[df23[23]==23]=1
df24[24]=df24['hour']
df24[df24[24]!= 24]=0
df24[df24[24]==24]=1

df1.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df2.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df3.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df4.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df5.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df6.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df7.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df8.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df9.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df10.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df11.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df12.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df13.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df14.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df15.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df16.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df17.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df18.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df19.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df20.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df21.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df22.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df23.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)
df24.drop(['date','demand','weekday','hour','day','year','month','temperature'],axis=1,inplace=True)

dataset=dataset.join([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22,df23,df24])
df=dataset

#Apagando colunas que podem enfregaquer o poder de  previsão do modelo
df.drop(['month','year','day','date','hour','weekday'],axis=1,inplace=True)

# Adicionando três novas colunas que irão melhorar na previsão de cargo a curto prazo
dhora_anterior = []
for i in range(len(df['demand'])):
    if i == 0:
        dhora_anterior.append(np.nan)
    else:
        dhora_anterior.append(df['demand'][i-1])

dhora_anterior = pd.DataFrame(dhora_anterior, columns = ['dhora_anterior'])


dhora_dia_anterior = []
for i in range(len(df['demand'])):
    if i<= 23:
        dhora_dia_anterior.append(np.nan)
    else:
        dhora_dia_anterior.append(df['demand'][i-24])

dhora_dia_anterior = pd.DataFrame(dhora_dia_anterior, columns = ['dhora_dia_anterior'])


dhora_semana_anterior = []
for i in range(len(df['demand'])):
    if i<= 167:
        dhora_semana_anterior.append(np.nan)
    else:
        dhora_semana_anterior.append(df['demand'][i-168])

dhora_semana_anterior = pd.DataFrame(dhora_semana_anterior, columns = ['dhora_semana_anterior'])


df = df.join([dhora_anterior, dhora_dia_anterior, dhora_semana_anterior])

# Separando os dados que serão usados para treinamento e para a previsão 
df['demand']=df['demand'].loc[:103271]
df['dhora_anterior']=df['dhora_anterior'].loc[:103272]
df['dhora_dia_anterior']=df['dhora_dia_anterior'].loc[:103295]
df1=df.loc[169:103271]
prev=df.loc[103272:103439]
x=df1.drop('demand',axis=1)
y=df1['demand']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

#modelo
model=Sequential()
model.add(Dense(250,input_shape=(35,),activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='linear'))
model.compile(loss='mean_squared_error',optimizer='adam')
l=model.fit(x_train,y_train,epochs=100,batch_size=50)

# Adicionando os valores faltantes das colunas da hora anterior,da hora do dia anteriro e da hora da semana anterior
prev1=prev.drop('demand',axis=1)

prev30=prev1.loc[103272:103295]
for i in range(103272,103295):
    x=prev30.loc[i]
    x=np.array(x)
    x=x.reshape(1,35)
    yi=model.predict(x)
    yi=int(yi)
    prev30['dhora_anterior'][i+1]=yi
    
prev40=prev1.loc[103296:103319]
h=prev30.loc[103295]
h=np.asarray(h).astype(np.float32)
h=h.reshape(1,35)
yi=model.predict(h)
yi=int(yi)
prev40['dhora_anterior'].loc[103296]=yi

n=prev30['dhora_anterior'].loc[103273:103295]
n=pd.Series(n)
prev40['dhora_dia_anterior'].loc[103296:103318]=n.values
prev40['dhora_dia_anterior'].loc[103319]=prev40['dhora_anterior'].loc[103296]
for i in range(103296,103319):
    x=prev40.loc[i]
    x=np.asarray(x).astype(np.float32)
    x=x.reshape(1,35)
    yi=model.predict(x)
    yi=int(yi)
    prev40['dhora_anterior'][i+1]=yi

prev50=prev1.loc[103320:103343]
h=prev40.loc[103319]
h=np.asarray(h).astype(np.float32)
h=h.reshape(1,35)
yi=model.predict(h)
yi=int(yi)
prev50['dhora_anterior'].loc[103320]=yi
prev50['dhora_dia_anterior'].loc[103343]=yi

n1=prev40['dhora_anterior'].loc[103297:103319]
n1=pd.Series(n1)
prev50['dhora_dia_anterior'].loc[103320:103342]=n1.values
for i in range(103320,103343):
    s=prev50.loc[i]
    s=np.asarray(s).astype(np.float32)
    s=s.reshape(1,35)
    wi=model.predict(s)
    wi=int(wi)
    prev50['dhora_anterior'][i+1]=wi

prev60=prev1.loc[103344:103367]
n2=prev50['dhora_anterior'].loc[103321:103343]
n2=pd.Series(n2)
prev60['dhora_dia_anterior'].loc[103344:103366]=n2.values
h=prev50.loc[103343]
h=np.asarray(h).astype(np.float32)
h=h.reshape(1,35)
yi=model.predict(h)
yi=int(yi)
prev60['dhora_anterior'].loc[103344]=yi
prev60['dhora_dia_anterior'].loc[103367]=yi

for i in range(103344,103367):
    s=prev60.loc[i]
    s=np.asarray(s).astype(np.float32)
    s=s.reshape(1,35)
    wi=model.predict(s)
    wi=int(wi)
    prev60['dhora_anterior'].loc[i+1]=wi

prev70=prev1.loc[103368:103391]
n3=prev60['dhora_anterior'].loc[103345:103367]
n3=pd.Series(n3)
prev70['dhora_dia_anterior'].loc[103368:103390]=n3.values
h=prev60.loc[103367]
h=np.asarray(h).astype(np.float32)
h=h.reshape(1,35)
yi=model.predict(h)
yi=int(yi)
prev70['dhora_anterior'].loc[103368]=yi
prev70['dhora_dia_anterior'].loc[103391]=yi

for i in range(103368,103391):
    s=prev70.loc[i]
    s=np.asarray(s).astype(np.float32)
    s=s.reshape(1,35)
    wi=model.predict(s)
    wi=int(wi)
    prev70['dhora_anterior'].loc[i+1]=wi

prev80=prev1.loc[103392:103415]
n4=prev70['dhora_anterior'].loc[103369:103391]
n4=pd.Series(n4)
prev80['dhora_dia_anterior'].loc[103392:103414]=n4.values
h=prev70.loc[103391]
h=np.asarray(h).astype(np.float32)
h=h.reshape(1,35)
yi=model.predict(h)
yi=int(yi)
prev80['dhora_anterior'].loc[103392]=yi
prev80['dhora_dia_anterior'].loc[103415]=yi
for i in range(103392,103415):
    s=prev80.loc[i]
    s=np.asarray(s).astype(np.float32)
    s=s.reshape(1,35)
    wi=model.predict(s)
    wi=int(wi)
    prev80['dhora_anterior'].loc[i+1]=wi

prev90=prev1.loc[103416:103439]
n5=prev80['dhora_anterior'].loc[103393:103415]
n5=pd.Series(n5)
prev90['dhora_dia_anterior'].loc[103416:103438]=n4.values
h=prev80.loc[103415]
h=np.asarray(h).astype(np.float32)
h=h.reshape(1,35)
yi=model.predict(h)
yi=int(yi)
prev90['dhora_anterior'].loc[103416]=yi
prev90['dhora_dia_anterior'].loc[103439]=yi
for i in range(103416,103439):
    s=prev90.loc[i]
    s=np.asarray(s).astype(np.float32)
    s=s.reshape(1,35)
    wi=model.predict(s)
    wi=int(wi)
    prev90['dhora_anterior'].loc[i+1]=wi


# Fazendo a previsão da semana escolhida
previsao=model.predict(prev1)

# Dados reais da semena prevista  do banco de dados
data=pd.read_csv('selected_data_ISONE.csv')
real=data['demand'].loc[103272:103439]

# métricas utilizadas
# r2_score
r2=r2_score(real,previsao)
# mape
previsao=np.array(previsao)
previsao=previsao.reshape(168)
mape=np.mean(np.abs((previsao - real) / real)) * 100

# Gráfico para analisar a previsão
real=np.array(real)
plt.plot(previsao,label='previsao')
plt.plot(real,label='real')
plt.legend()
plt.xlabel('dias')
plt.ylabel('demanda')
plt.show()
