import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import metrics

digits = load_digits()

plt.figure(figsize=(20,4))
for index,(image,label) in enumerate (zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1,5,index+1)
    plt.imshow(np.reshape(image,(8,8)), cmap=plt.cm.gray)
    plt.title("Training: %i\n" %label, fontsize=20)
plt.show()

X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target, test_size=0.23,random_state=2)

from sklearn.linear_model import LogisticRegression
logireg = LogisticRegression()
logireg.fit(X_train,y_train)

print(logireg.predict(X_test[0].reshape(1,-1)))

prediction = logireg.predict(X_test)

score = logireg.score(X_test,y_test)

cm = metrics.confusion_matrix(y_test,prediction)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True)
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
all_sample_title = "Accuracy score: {0}".format(score)
plt.title(all_sample_title, size = 15)

plt.show()

index = 0
misclassifiedIndex = []
for predict, actual in zip(prediction,y_test):
    if predict == actual:
        misclassifiedIndex.append(index)
    index+=1
plt.figure(figsize = (20,3))
for plotIndex, wrong in enumerate(misclassifiedIndex[0:4]):
    plt.subplot(1,4, plotIndex+1)
    plt.imshow(np.reshape(X_test[wrong],(8,8)), cmap=plt.cm.gray)
    plt.title("predicted: {}, actual: {}".format(prediction[wrong], y_test[wrong]), fontsize=20)
    
plt.show()
    

