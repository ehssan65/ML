import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# داده‌های نمونه
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # متغیر مستقل (ویژگی)
y = np.array([2, 3.5, 5.8, 5, 4.6])  # متغیر وابسته

# ایجاد یک شیء از کلاس مدل رگرسیون خطی
model = LinearRegression()

# آموزش مدل با داده‌های نمونه
model.fit(X, y)

# پیش‌بینی مقادیر متغیر وابسته برای داده‌های جدید
X_new = np.array([12,20,16,45]).reshape(-1, 1)
y_pred = model.predict(X_new)

# نمایش نتایج
print("پیش‌بینی مقادیر جدید:", y_pred)

# رسم مدل
plt.scatter(X, y, color='blue')  # نقاط داده
plt.plot(X, model.predict(X), color='red')  # خط رگرسیون
plt.title('رگرسیون خطی')
plt.xlabel('متغیر مستقل (ویژگی)')
plt.ylabel('متغیر وابسته')
plt.show()
