#kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

#2.veri ön işleme
#2.1 veri yükleme
veriler=pd.read_csv("data.csv")     

boy=veriler[['boy']]
print(boy)
boykilo=veriler[["boy","kilo"]]
print(boykilo)

     
#eksik veriler
eksikveriler=pd.read_csv("eksikveriler.csv")
print(eksikveriler)


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
yas=eksikveriler.iloc[:,1:4].values
print(yas)
imputer=imputer.fit(yas[:,1:4])
yas[:,1:4]=imputer.transform(yas[:,1:4])
print(yas)
#encoder-kategorik değerlerden nümerik verilere dönüşüm
ulke=veriler.iloc[:,0:1].values
print(ulke)
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke)
#ohe-kolon başlıklarının etiketleri
ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)
#numpy dizileri dataframe dönüşümü
sonuc=pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])
print(sonuc)
sonuc2=pd.DataFrame(data=yas,index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)
cinsiyet=veriler.iloc[:,-1].values
print(cinsiyet)
sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
print(sonuc3)
#veribirleştirme-dataframe birleştirme işlemi
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)
#verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)
#öznitelikölçekleme-verilerin ölçeklendirmesi-verileri standartlaştırma
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)







































