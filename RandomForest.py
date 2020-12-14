import numpy as np
import csv
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve

from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score



# this disctories are used to store the features
dict = {}
dict1 = {}
dict2 = {}
dict3 = {}
dict4 = {}
y = []
y1 = []
y2 = []


# train
# here Hub1.csv is our csv file. To make this code working change the file path
with open('Hub1.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    row_num = 0
    for row in readCSV:
        print(row)
        # 80% data for training
        if (row_num != 0 and row_num < 2305):
            # Class level is stored in col 7
            y.append(int(row[7]))

            x = []

            x2 = []
            for col in range(0, 6):
                # print new

                x.append(row[col])
                # print row[col]
            dict[row_num] = x

            # Also test for centrality measure degree
            x2.append(row[0])
                # print row[col]
            dict3[row_num] = x2

        # 20% data for testing
        elif (row_num != 0 and row_num >= 2305):

            y1.append(int(row[7]))

            x1 = []
            for col in range(0, 6):
                # print new

                x1.append(row[col])
                # print row[col]
            dict1[row_num - 2305 + 1] = x1

            x3 = []


            x3.append(row[0])
                # print row[col]
            dict4[row_num - 2305 + 1] = x3

        row_num += 1
#Create the feature vectors
X = []
for i in range(1, len(dict) + 1):
    new = dict[i]
    X.append(new)
X1 = []
for i in range(1, len(dict1) + 1):
    new = dict1[i]
    X1.append(new)
X2 = []
for i in range(1, len(dict3) + 1):
    new = dict3[i]
    X2.append(new)
X3 = []
for i in range(1, len(dict4) + 1):
    new = dict4[i]
    X3.append(new)
print(X)
print(y)
print(X1)
print(y1)

clf=RandomForestClassifier(n_estimators=40)
clf1=RandomForestClassifier(n_estimators=40)
#Train the model using the training sets y_pred=clf.predict(X_test)
#Fit. clf is for 6 centrality measure and clf1 for degree centrality alone
clf.fit(X,y)
clf1.fit(X2,y)
y_pred1=clf1.predict(X3)
y_pred=clf.predict(X1)

fpr_rf, tpr_rf, _ = roc_curve(y1, y_pred)
fpr_rf1, tpr_rf1, _ = roc_curve(y1, y_pred1)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')

#plot ROC curves
plt.plot(fpr_rf, tpr_rf, label='Random forest with six centrality')
plt.plot(fpr_rf1, tpr_rf1, label='Random forest with only degree centrality')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

#model accuracy
print("Accuracy:",metrics.accuracy_score(y1, y_pred))
print("Accuracy Degree:",metrics.accuracy_score(y1, y_pred1))
rfc_cv_score = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
print("=== Confusion Matrix ===")
print(confusion_matrix(y1, y_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(y1, y_pred))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())

#Calculate feature importance
importance = clf.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
import seaborn as sns
sns.barplot(x=importance, y=[x for x in range(len(importance))])
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


average_precision = average_precision_score(y1, y_pred)
disp = plot_precision_recall_curve(clf, X1, y1)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
plt.show()