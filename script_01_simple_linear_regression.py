
# In[ ]: Importação dos pacotes necessários
    
import pandas as pd # manipulação de dado em formato de dataframe
import seaborn as sns # biblioteca de visualização de informações estatísticas
import matplotlib.pyplot as plt # biblioteca de visualização de dados
import statsmodels.api as sm # biblioteca de modelagem estatística
import numpy as np # biblioteca para operações matemáticas multidimensionais
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
import plotly.graph_objs as go # gráfico 3D
from scipy.stats import pearsonr # correlações de Pearson
from sklearn.preprocessing import LabelEncoder # transformação de dados


# In[ ]:
#############################################################################
#                          REGRESSÃO LINEAR SIMPLES                         #
#                  EXEMPLO 01 - CARREGAMENTO DA BASE DE DADOS               #
#############################################################################
    
df = pd.read_csv('data-raw/Salary_dataset.csv', delimiter=',')
df

#Características das variáveis do dataset
df.info()

#Estatísticas univariadas
df.describe()


# In[ ]: Gráfico de dispersão

#Regressão linear que melhor se adequa às obeservações: função 'sns.lmplot'

plt.figure(figsize=(20,10))
sns.lmplot(data=df, x='YearsExperience', y='Salary', ci=False)
plt.xlabel('YearsExperience', fontsize=20)
plt.ylabel('Salary', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=14)
plt.show


# In[ ]: Estimação do modelo de regressão linear simples

#Estimação do modelo
modelo = sm.OLS.from_formula('Salary ~ YearsExperience', df).fit()

#Observação dos parâmetros resultantes da estimação
modelo.summary()


# In[ ]: Salvando fitted values (variável yhat) 
# e residuals (variável erro) no dataset

df['yhat'] = modelo.fittedvalues
df['erro'] = modelo.resid
df


# In[ ]: Gráfico didático para visualizar o conceito de R²

y = df['Salary']
yhat = df['yhat']
x = df['YearsExperience']
mean = np.full(x.shape[0] , y.mean(), dtype=int)

for i in range(len(x)-1):
    plt.plot([x[i],x[i]], [yhat[i],y[i]],'--', color='#2ecc71')
    plt.plot([x[i],x[i]], [yhat[i],mean[i]], ':', color='#9b59b6')
    plt.plot(x, y, 'o', color='#2c3e50')
    plt.axhline(y = y.mean(), color = '#bdc3c7', linestyle = '-')
    plt.plot(x,yhat, color='#34495e')
    plt.title('R2: ' + str(round(modelo.rsquared,4)))
    plt.xlabel("YearsExperience")
    plt.ylabel("Salary")
    plt.legend(['Erro = Y - Yhat', 'Yhat - Ymédio'], fontsize=10)
plt.show()


# In[ ]: Cálculo manual do R²

R2 = ((df['yhat']-
       df['Salary'].mean())**2).sum()/(((df['yhat']-
                                        df['Salary'].mean())**2).sum()+
                                        (df['erro']**2).sum())

round(R2,4)


# In[ ]: Coeficiente de ajuste (R²) é a correlação ao quadrado

#Correlação de Pearson
df[['Salary','YearsExperience']].corr()

#R²
(df[['Salary','YearsExperience']].corr())**2

#R² de maneira direta
modelo.rsquared



# In[ ]: Voltando ao nosso modelo original

#Plotando o intervalo de confiança de 90%
plt.figure(figsize=(20,10))
sns.regplot(data=df, x='YearsExperience', y='Salary', ci=90, color='purple')
plt.xlabel('YearsExperience', fontsize=20)
plt.ylabel('Salary', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24)
plt.show



# In[ ]: Calculando os intervalos de confiança

#Nível de significância de 10% / Nível de confiança de 90%
modelo.conf_int(alpha=0.1)

#Nível de significância de 5% / Nível de confiança de 95%
modelo.conf_int(alpha=0.05)

#Nível de significância de 1% / Nível de confiança de 99%
modelo.conf_int(alpha=0.01)

#Nível de significância de 0,001% / Nível de confiança de 99,999%
modelo.conf_int(alpha=0.00001)


# In[ ]: Fazendo predições em modelos OLS
#Ex.: Qual seria o tempo gasto, em média, para percorrer a distância de 25km?

modelo.predict(pd.DataFrame({'YearsExperience':[25]}))


