from sklearn import datasets
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

diabetes = datasets.load_diabetes()
# print('diabetes', diabetes)
diabetes_x = diabetes.data[:, np.newaxis]
# print('diabetes_x', diabetes_x)
diabetes_x_temp = diabetes_x[:, :, 2]
# print('diabetes_x_temp', diabetes_x_temp)

diabetes_x_train = diabetes_x_temp[:-20]
# print('diabetes_x_train', diabetes_x_train)
diabetes_x_test = diabetes_x_temp[-20:]
# print('diabetes_x_test', diabetes_x_test)
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# print(u'划分行数:', len(diabetes_x_temp), len(diabetes_x_train), len(diabetes_x_test))
# print(diabetes_x_test)
clf = linear_model.LinearRegression()
clf.fit(diabetes_x_train, diabetes_y_train)

# print('Coefficients :\n', clf.coef_)
# print('Residual sum of square: %.2f,'%np.mean(clf.predict(dia)))

plt.title(u'LinearRegression Diabetes')   #标题
plt.xlabel(u'Attributes')                 #x轴坐标
plt.ylabel(u'Measure of disease')         #y轴坐标

plt.scatter(diabetes_x_test, diabetes_y_test, color='black')
plt.plot(diabetes_x_test, clf.predict(diabetes_x_test), color='blue', linewidth=3)
plt.show()

