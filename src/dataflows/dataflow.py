import pandas as pd

from src.components.classification import rf_classification, pred_report
from src.components.feature_engineering import calc_mean, calc_count, join_dfs, merge_calc_cols, split_data, \
    plot_2d_space, implement_oversampling, scale_features, reduce_dimensions, split_new_data, scale_features_new_data, \
    reduce_dimensions_new_data


def feature_engineering(olist_orders, olist_order_items, olist_products, olist_order_payments, olist_order_reviews,
                        olist_customers):
    churn = olist_orders[olist_orders['order_status'].isin(['canceled', 'delivered'])]
    churn = churn.drop(columns=['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date',
                                'order_estimated_delivery_date'])

    # join in ald calculate needed columns from different dataframes
    olist_order_items = join_dfs(olist_order_items, olist_products, join_key=['product_id'], drop_nan=['product_id'])
    count_order_items = calc_count(olist_order_items[['order_id', 'product_id']], groupkey=['order_id'],
                                   dict={'product_id': 'count_order_items'})
    avg_calc_order_items = calc_mean(olist_order_items[[
        'order_id', 'price', 'freight_value', 'product_name_lenght', 'product_description_lenght',
        'product_name_lenght',
        'product_photos_qty',
        'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']],
                                     groupkey=['order_id'], dict={
            'price': 'avg_price',
            'freight_value': 'avg_freight_value',
            'product_name_lenght': 'avg_product_name_length',
            'product_description_lenght': 'avg_product_description_length',
            'product_photos_qty': 'avg_product_photos_qty',
            'product_weight_g': 'avg_product_weight_g',
            'product_length_cm': 'avg_product_length_cm',
            'product_height_cm': 'avg_product_height_cm',
            'product_width_cm': 'avg_product_width_cm'})

    olist_order_payments_reduced = olist_order_payments[['order_id', 'payment_type']]

    count_calc_reviews = calc_count(olist_order_reviews[['order_id', 'review_comment_message', 'review_score']],
                                    groupkey='order_id', dict={
            'review_comment_message': 'count_product_comments',
            'review_score': 'count_product_ratings'})

    avg_calc_ratings = calc_mean(olist_order_reviews[['order_id', 'review_score']], groupkey='order_id',
                                 dict={'review_score': 'avg_rating'})

    df_to_merge = [churn, count_order_items, avg_calc_order_items, olist_order_payments_reduced, count_calc_reviews,
                   avg_calc_ratings]
    churn = merge_calc_cols(df_to_merge, join_key=['order_id'])

    churn = join_dfs(churn, olist_customers[['customer_id', 'customer_state', 'customer_city']],
                     join_key=['customer_id'],
                     drop_nan=['customer_id'])

    # set order_id as index
    churn = churn.set_index('order_id').dropna().drop(
        columns=['order_purchase_timestamp', 'customer_id', 'customer_city']).drop_duplicates()

    # transform to categorical dtype
    churn['order_status'] = churn['order_status'].astype('category')
    churn['customer_state'] = churn['customer_state'].astype('category')
    churn['payment_type'] = churn['payment_type'].astype('category')

    # get dummies from categorical columns
    churn = pd.get_dummies(data=churn, columns=['order_status', 'customer_state', 'payment_type']).drop(
        columns=['order_status_delivered'])

    # split data via train_test_split
    X, y, X_train, X_test, y_train, y_test = split_data(churn)

    # scale features (normalize data)
    X_train, X_test = scale_features(X_train, X_test)

    # reduce dimensions via PCA
    X_train, X_test, explained_variance = reduce_dimensions(X_train, X_test)

    # visualize imbalanced data
    plot_2d_space(X, y, 'Imbalanced dataset (2 PCA components)')

    # visualize data after oversampling
    X_ros, y_ros = implement_oversampling(X, X_train, y_train)
    plot_2d_space(X_ros, y_ros, 'Random over-sampling')
    return X_ros, y_ros, X_test, y_test


def rf_classification_model(X_ros, y_ros):
    # Apply random forest as classification method
    rf_model = rf_classification(X_ros, y_ros)
    return rf_model


def rf_prediction(rf_model, X, y):
    # Prediction and classification report including Accuracy and F1 score
    rf_report = pred_report(rf_model, X, y)
    return rf_report


def feature_engineering_new_data(olist_orders, olist_order_items, olist_products, olist_order_payments,
                                 olist_order_reviews,
                                 olist_customers):
    churn = olist_orders[olist_orders['order_status'].isin(['canceled', 'delivered'])]
    churn = churn.drop(columns=['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date',
                                'order_estimated_delivery_date'])

    # join in ald calculate needed columns from different dataframes
    olist_order_items = join_dfs(olist_order_items, olist_products, join_key=['product_id'], drop_nan=['product_id'])
    count_order_items = calc_count(olist_order_items[['order_id', 'product_id']], groupkey=['order_id'],
                                   dict={'product_id': 'count_order_items'})
    avg_calc_order_items = calc_mean(olist_order_items[[
        'order_id', 'price', 'freight_value', 'product_name_lenght', 'product_description_lenght',
        'product_name_lenght',
        'product_photos_qty',
        'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']],
                                     groupkey=['order_id'], dict={
            'price': 'avg_price',
            'freight_value': 'avg_freight_value',
            'product_name_lenght': 'avg_product_name_length',
            'product_description_lenght': 'avg_product_description_length',
            'product_photos_qty': 'avg_product_photos_qty',
            'product_weight_g': 'avg_product_weight_g',
            'product_length_cm': 'avg_product_length_cm',
            'product_height_cm': 'avg_product_height_cm',
            'product_width_cm': 'avg_product_width_cm'})

    olist_order_payments_reduced = olist_order_payments[['order_id', 'payment_type']]

    count_calc_reviews = calc_count(olist_order_reviews[['order_id', 'review_comment_message', 'review_score']],
                                    groupkey='order_id', dict={
            'review_comment_message': 'count_product_comments',
            'review_score': 'count_product_ratings'})

    avg_calc_ratings = calc_mean(olist_order_reviews[['order_id', 'review_score']], groupkey='order_id',
                                 dict={'review_score': 'avg_rating'})

    df_to_merge = [churn, count_order_items, avg_calc_order_items, olist_order_payments_reduced, count_calc_reviews,
                   avg_calc_ratings]
    churn = merge_calc_cols(df_to_merge, join_key=['order_id'])

    churn = join_dfs(churn, olist_customers[['customer_id', 'customer_state', 'customer_city']],
                     join_key=['customer_id'],
                     drop_nan=['customer_id'])

    # set order_id as index
    churn = churn.set_index('order_id').dropna().drop(
        columns=['order_purchase_timestamp', 'customer_id', 'customer_city']).drop_duplicates()

    # transform to categorical dtype
    churn['order_status'] = churn['order_status'].astype('category')
    churn['customer_state'] = churn['customer_state'].astype('category')
    churn['payment_type'] = churn['payment_type'].astype('category')

    # get dummies from categorical columns
    churn = pd.get_dummies(data=churn, columns=['order_status', 'customer_state', 'payment_type']).drop(
        columns=['order_status_delivered'])

    # split data via train_test_split
    X, y = split_new_data(churn)

    # scale features (normalize data)
    X = scale_features_new_data(X)

    # reduce dimensions via PCA
    X, explained_variance = reduce_dimensions_new_data(X)
    return X, y


def pred_new_data(model, X):
    y_pred_new = model.predict(X)
    return y_pred_new