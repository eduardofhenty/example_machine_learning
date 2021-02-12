# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 10:40:33 2021

@author: Eduardo
"""
#Baseado em https://www.linkedin.com/pulse/implementando-um-árvore-de-decisão-com-python-ironia-e-faria/


#ler dados do INMET

import pandas as pd
from sklearn.tree import DecisionTreeClassifier,export_graphviz,plot_tree
import graphviz 

#Função para visualizar os resultados e atualizar o dataframe
def visualize_res(df_res):
    df_res= df_res.assign(geada=0)    
    num_lin= (len(df_res.index))        
            
    
    for i in range(0,num_lin,1):           
        
        if  df_res.loc[i,['fraca']].to_numpy() ==1:
            print(df_res.loc[i,['data']]+ ' fraca')
            df_res.loc[i,['geada']]='Fraca'
        elif df_res.loc[i,['moderada']].to_numpy()==1:
            print(df_res.loc[i,['data']]+ ' moderada')
            df_res.loc[i,['geada']]='Moderada'     
        elif df_res.loc[i,['forte']].to_numpy()==1:
            print(df_res.loc[i,['data']]+ ' forte')
            df_res.loc[i,['geada']]='forte'     
        else:
            df_res.loc[i,['geada']]='Sem geada'  
            
    return df_res
    
    
    
    

#Santa Vitoria do Palmar Estação automatica. Ano 2017
#Estação com um boa ocorrencia de geadas
file = "C:/Users/Eduardo/Desktop/machine learning curso/exemplos/data/geada/dados_A899_D_2017-01-01_2017-12-31.csv"

df = pd.read_csv(file, delimiter=";",skiprows=10,decimal=".",   header=0,na_filter=True)

df = df[['Data Medicao',         
         'TEMPERATURA MAXIMA, DIARIA (AUT)(°C)',
         'TEMPERATURA MEDIA, DIARIA (AUT)(°C)',
         'TEMPERATURA MINIMA, DIARIA (AUT)(°C)',
         'UMIDADE RELATIVA DO AR, MEDIA DIARIA (AUT)(%)',         
         'VENTO, VELOCIDADE MEDIA DIARIA (AUT)(m/s)'        
        ]]

df.columns=['data','t_max','t_med','t_min','ur','vento']
df_treino = df

#Santa Maria #Estação convenciaonal. Ano 2017
#Ocorrencias de geada
#20/06/2017 	1ºC 	Moderada
#20/07/2017 	0.8ºC 	Forte
#19/07/2017 	-1.2ºC 	Forte
#18/07/2017 	-0.2ºC 	Forte
file = "C:/Users/Eduardo/Desktop/machine learning curso/exemplos/data/geada/dados_83936_D_2017-01-01_2017-12-31.csv"

df = pd.read_csv(file, delimiter=";",skiprows=10,decimal=".",   header=0,na_filter=True)


df= df[ ['Data Medicao', 'TEMPERATURA MAXIMA, DIARIA(°C)', 
     'TEMPERATURA MINIMA, DIARIA(°C)',
     'TEMPERATURA MEDIA COMPENSADA, DIARIA(°C)',
     'UMIDADE RELATIVA DO AR, MEDIA DIARIA(%)',
     'VENTO, VELOCIDADE MEDIA DIARIA(m/s)'
     ] ]


df.columns=['data','t_max','t_min','t_med','ur','vento']

df_teste =df

#Dados de geada. Santa Vitoria do Palmar. Ano 2017
file = "C:/Users/Eduardo/Desktop/machine learning curso/exemplos/data/geada/geada 2017 rsb.csv"

df_geada = pd.read_csv(file, delimiter=";",decimal=".",   header=0)

fraca=[]
moderada=[]
forte=[]
lista_geada = df_geada['Intensidade']

i=0
for dia in df_treino['data']:
    aux=True
 
    for diageada in df_geada['data']:
        if (dia==diageada):
            
            if lista_geada[i]=='Fraca':
                #print('Fraca')
                fraca.append(1)
                moderada.append(0)
                forte.append(0)
                
            if lista_geada[i]=='Moderada':
                fraca.append(0)
                moderada.append(1)
                forte.append(0)
       
            if lista_geada[i]=='Forte':
                fraca.append(0)
                moderada.append(0)
                forte.append(1)
            aux=False  
            i=i+1 
           
    if(aux):
        fraca.append(0)
        moderada.append(0)
        forte.append(0)
    
            
d ={'fraca':fraca,'moderada':moderada,'forte':forte }
            
df_geada2=  pd.DataFrame(data=   d  )


#Juntando os dataframe

df_treino_f = pd.concat([df_treino,df_geada2],   axis=1)


#Seperando os dados de entrada e saída para o treinamento
X_treino= df_treino_f [['t_max','t_min','t_med','ur','vento']]


y_treino = df_treino_f[['fraca','moderada','forte']]
clf = DecisionTreeClassifier()
clf = clf.fit( X_treino,y_treino)

 

#Aplicando nos dados de avaliação
X_teste = df_teste[['t_max','t_min','t_med','ur','vento'] ]
res= clf.predict(X_teste)

#Imprimindo os resultados

df_res= pd.concat([df_teste[['data']], pd.DataFrame( res,columns=['fraca','moderada','forte'])] ,axis=1) 

df_res= visualize_res(df_res)

plot_tree(clf,feature_names=['t_max','t_min','t_med','ur','vento'],                      
                      filled=True, rounded=True,fontsize=6.0
                    )

dot_data = export_graphviz( 
         clf, 
         out_file=None,
         feature_names=['t_max','t_min','t_med','ur','vento'],         
         #class_names=['fraca',"n",'moderada',"n",'forte',"n"],  
         class_names=True,
         filled=True, rounded=True,
         proportion=True,
         node_ids=True,
         rotate=False,
         label='all',
         special_characters=True
        )  
graph = graphviz.Source(dot_data)  
#Salva a árvore em um pdf
graph.render("geada")