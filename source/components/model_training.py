## Get X and y
X = churn.drop(columns = ['is_canceled', 'order_hour_minute_second', 'order_year_month_day']).values
y = churn['is_canceled'].values


# Train-Test-Split (stratified shuffle split)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y, shuffle = True)


# Feature Scaling (normalizing the data)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Feature Engineering -  PCA for dimension reduction (Real coordinate space: metric/ binary)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_


# visualize imbalanced data
def plot_2d_space(X, y, label='Classes'):
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()

plot_2d_space(X, y, 'Imbalanced dataset (2 PCA components)')


#implement oversampling
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()
X_ros, y_ros = ros.fit_sample(X_train, y_train)

print(X_ros.shape[0] - X.shape[0], 'new random picked points')

plot_2d_space(X_ros, y_ros, 'Random over-sampling')




##from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

##lda = LDA(n_components=1)
##X_train = lda.fit_transform(X_train, y_train)
##X_test = lda.transform(X_test)


## Apply random forest as classification method
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0)

#classifier.fit(X_train, y_train)
classifier.fit(X_ros, y_ros)

y_pred = classifier.predict(X_test)


# evaluate prediction performance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy' + str(accuracy_score(y_test, y_pred)))

## f1 score to interpret
from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred, average="macro"))
