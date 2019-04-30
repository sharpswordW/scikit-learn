import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

X = [[6], [8], [10], [14], [18]]
Y = [[7], [9], [13], [17.5], [18]]

clf = linear_model.LinearRegression()
clf.fit(X, Y)
res = clf.predict(np.array([12]).reshape(-1, 1))[0]
print('预测一张12英寸匹萨价格：$%.2f', res)

print(u'系数:', clf.coef_)
print(u'截距:', clf.intercept_)
print(u'评分系数:', clf.score(X, Y))

X2 = [[0], [10], [12], [13], [14], [25], [100]]
Y2 = clf.predict(X2)
print('Y2:', Y2)

plt.figure()
plt.title(u'diameter-cost curver')
plt.xlabel(u'diameter')
plt.ylabel(u'cost')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.plot(X, Y, 'k.')
plt.plot(X2, Y2, 'g-')
plt.show()









