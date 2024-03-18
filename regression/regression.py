import numpy as np
import matplotlib.pyplot as plt   
from  sklearn.linear_model import LinearRegression




'''
comment construire un model

model = nom du constructeur


entrainer 

model.fit(x,y)          //x y  donnee a deux dimension

evaluer 
model.score(x,y)

prediction
model.predict(x)

'''


np.random.seed(0)
m = 100
x = np.linspace(0,10,m).reshape(m,1)
y = x + np.random.randn(m, 1)
plt.scatter(x,y)
model = LinearRegression()

model.fit(x,y)

model.score(x,y)

predictions = model.predict(x)

plt.scatter(x,y)
plt.plot(x,predictions,c = 'r')
