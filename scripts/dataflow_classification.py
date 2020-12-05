from source.components.classification import rf_classification, pred_report

# Apply random forest as classification method
rf_model = rf_classification(X_ros, y_ros)

# Prediction and classification report including Accuracy and F1 score
rf_report = pred_report(rf_model, X_test, y_test)
