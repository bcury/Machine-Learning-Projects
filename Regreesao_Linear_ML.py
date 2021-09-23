#REGRESSÃO LINEAR

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


Base_Dados = pd.read_excel('C:/Users/bruno/Desktop/Estudos/Projects/Dados/BaseDados_RegressaoLinear.xlsx',engine ='openpyxl')
print(Base_Dados)
Base_Dados.head(5)
Base_Dados.tail()
Base_Dados.describe()

#converter para um array

Eixox = Base_Dados.iloc[:,0].values
Eixoy = Base_Dados.iloc[:,1].values
print(Eixox)
print(Eixoy)

#coisa errada com a importação do matplotlib

plt.figure(figsize=(10,5))
plt.scatter(Eixox, Eixoy) ;
plt.title('Gráfico com 2 eixos [Salário X Limite]')
plt.xlabel('Salario')
plt.ylabel('limite');

sns.heatmap(Base_Dados.isnull());
sns.pairplot(Base_Dados);

Correlação = np.corrcoef(Eixox,Eixoy)
print(Correlação)
plt.figure(figsize=(10,5))
sns.heatmap(Correlação, annot=True);

#deixando em formato de matriz, anterior estava em tipo lista

Eixox = Eixox.reshape(-1,1)
Eixoy = Eixoy.reshape(-1,1)

#Dados de teste e treino - ML

from sklearn.model_selection import train_test_split

x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(
    Eixox,
    Eixoy,
    test_size = 0.20
)

print(len(x_treinamento), len(x_teste))

#agora implementando a Regressao linear no modelo ML

from sklearn.linear_model import LinearRegression

Funcao_Regressao = LinearRegression()
Funcao_Regressao.fit(x_treinamento, y_treinamento)  #aplica calculo estatisticos pra treinar o modelo de ML
Funcao_Regressao.score(x_treinamento,y_treinamento) #o quanto o modelo se aproximou, tipo accurate

plt.figure(figsize=(10,5))
plt.scatter(x_treinamento,y_treinamento)
plt.plot(x_teste,Funcao_Regressao.predict(x_teste),color='red')

#AVALIAR O desemepnho do modelo, como erro, etc

Previsoes = Funcao_Regressao.predict(x_teste)
from sklearn import metrics
print('RMSE', np.sqrt(metrics.mean_squared_error(y_teste, Previsoes)))  #erro medio da nossa regressao


print(Funcao_Regressao.predict([[5600]]))  # vc coloca o salario e a resposta é o emprestimo que a pessoa recebe devido ao salario










