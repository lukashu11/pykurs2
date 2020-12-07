from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def rf_classification(X_ros, y_ros):
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    model = classifier.fit(X_ros, y_ros)
    return model


def pred_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print('Accuracy' + str(accuracy_score(y_test, y_pred)))
    print(f1_score(y_test, y_pred, average="macro"))



