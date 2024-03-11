import re
import math
import numpy as np
import pandas as pd
import random
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import svm
#fo= open("COutput.txt","w+")
#Correct code
str2 = set()
ult = []
ult1= []
dd= []
cols_to_use = ["text","Type"]
DD = pd.read_csv("True.csv",usecols= cols_to_use)
DD1 = pd.read_csv("Fake.csv",usecols= cols_to_use)
print (len(DD),len(DD1))
frames = [DD,DD1]
result = pd.concat(frames)
result = result.sample(frac=1).reset_index(drop=True)
result = result.head(35000)
print(result)
result['label'] = result['Type'].apply(lambda x: 0 if x=='Fake' else 1)
X_train, X_test, y_train, y_test = train_test_split(result['text'], result['label'], random_state=1)
cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
X_train_cv = cv.fit_transform(X_train.values.astype('U'))
X_test_cv = cv.transform(X_test.values.astype('U'))
print(X_train_cv)


Big = []
for it in range(1,10):
    avg = 0
    clf = svm.SVC(C=(.2*it),kernel= "linear")
    y_pred5 = cross_val_score(clf, X_train_cv, y_train, cv=3)
    for ku in y_pred5:
        avg = avg +ku
        print(ku)
    avg = avg/3
    Big.append(avg)








#Naive bayes
for i in range(1,11,2):
    avg = 0
    naive_bayes = MultinomialNB(alpha = i,fit_prior = True)
    naive_bayes.fit(X_train_cv, y_train)
    predictions = naive_bayes.predict(X_test_cv)
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    print('Accuracy score: ', accuracy_score(y_test, predictions))
    print('Precision score: ', precision_score(y_test, predictions))
    print('Recall score: ', recall_score(y_test, predictions))
    def get_score(model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        return model.score(X_test, y_test)
    scores = []
    folds = StratifiedKFold(n_splits=5)
    X_train_cvv = pd.DataFrame(X_train_cv).to_numpy()
    y_pred = cross_val_score(naive_bayes, X_train_cv, y_train, cv=10)

    for k in y_pred:
        avg = avg +k

    Big.append(avg/10)

#KNN
for ui in range(11,20):
    avg = 0
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=ui,metric = "euclidean")
    X_train_cvv = pd.DataFrame(X_train_cv).to_numpy()
    y_pred1 = cross_val_score(neigh, X_train_cv, y_train, cv=10)
    for hy in y_pred1:
        avg = avg+hy
        print(hy)
    avg = avg/10
    Big.append(avg)
for r in Big:
    print("Big",r)
import numpy as np
import matplotlib.pyplot as plt

# set height of bar
# length of these lists determine the number
# of groups (they must all be the same length)
bars1 = [Big[0]]
bars2 = [Big[1]]
bars3 = [Big[2]]
bars4 = [Big[3]]
bars5 = [Big[4]]
barWidth = 0.25
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]

# Make the plot
plt.bar(r1, bars1, color='green', width=barWidth, edgecolor='white', label='C = .2')
plt.bar(r2, bars2, color='red', width=barWidth, edgecolor='white', label='C = .4')
plt.bar(r3, bars3, color='black', width=barWidth, edgecolor='white', label='C = .6')
plt.bar(r4, bars4, color='blue', width=barWidth, edgecolor='white', label='C = .8')
plt.bar(r5, bars5, color='cyan', width=barWidth, edgecolor='white', label='C= 1.0')
# Add xticks on the middle of the group bars
plt.xlabel('Type of Model', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['SVM Accuracy '])

# Create legend & Show graphic
plt.legend()
plt.show()
plt.savefig("barChart.pdf",dpi=400,bbox_inches='tight',pad_inches=0.05) # save as a pdf

bars1 = [Big[0]]
bars2 = [Big[1]]
bars3 = [Big[2]]
bars4 = [Big[3]]
bars5 = [Big[4]]
barWidth = 0.25
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]

# Make the plot
plt.bar(r1, bars1, color='green', width=barWidth, edgecolor='white', label='K = 1')
plt.bar(r2, bars2, color='red', width=barWidth, edgecolor='white', label='K = 3')
plt.bar(r3, bars3, color='black', width=barWidth, edgecolor='white', label='K = 5')
plt.bar(r4, bars4, color='blue', width=barWidth, edgecolor='white', label='K = 7')
plt.bar(r5, bars5, color='cyan', width=barWidth, edgecolor='white', label='K = 9')
# Add xticks on the middle of the group bars
plt.xlabel('Type of Model', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['KNN Accuracy'])

# Create legend & Show graphic
plt.legend()
plt.show()
plt.savefig("barChart.pdf",dpi=400,bbox_inches='tight',pad_inches=0.05) # save as a pdf

""" Use these to test the data"""
neigh = KNeighborsClassifier(n_neighbors=ui,metric = "euclidean")
naive_bayes = MultinomialNB(alpha = i,fit_prior = True)
clf = svm.SVC(C=1.8,kernel= "linear")
