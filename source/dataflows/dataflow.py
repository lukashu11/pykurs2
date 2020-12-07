import pandas as pd
from source.components.feature_engineering import calc_mean, calc_count, join_dfs, merge_calc_cols, split_data, \
    plot_2d_space, implement_oversampling
from source.components.classification import rf_classification, pred_report


def feature_engineering(olist_orders, olist_order_items, olist_products, olist_order_payments, olist_order_reviews,
                        olist_customers):
    churn = olist_orders[olist_orders['order_status'].isin(['canceled', 'delivered'])]
    churn = churn.drop(columns=['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date',
                                'order_estimated_delivery_date'])

    # join in ald calculate needed columns from different dataframes
    olist_order_items = join_dfs(olist_order_items, olist_products, join_key=['product_id'], drop_nan=['product_id'])
    count_order_items = calc_count(olist_order_items[['order_id', 'product_id']], groupkey=['order_id'],
                                   dict={'product_id': 'count_order_items'})

    #TODO: Diese code-zeile erschlägt mich:
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

    #TODO: Bitte als ein Befehl schreiben!
    # Convert the data type of column Age to float64 & data type of column Marks to string
    # empDfObj = empDfObj.astype({'Age': 'float64', 'Marks': 'object'})
    # https://thispointer.com/pandas-change-data-type-of-single-or-multiple-columns-of-dataframe-in-python/

    # transform to categorical dtype
    churn['order_status'] = churn['order_status'].astype('category')
    churn['customer_state'] = churn['customer_state'].astype('category')
    churn['payment_type'] = churn['payment_type'].astype('category')

    # get dummies from categorical columns
    churn = pd.get_dummies(data=churn, columns=['order_status', 'customer_state', 'payment_type']).drop(
        columns=['order_status_delivered'])

    # normalize data, reduce to two dimensions via PCA and get train_test_split
    X, y, X_train, X_test, y_train, y_test, explained_variance = split_data(churn)



    # Get X and y
    X = df.drop(columns=['order_status_canceled']).values
    y = df['order_status_canceled'].values

    # train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y, shuffle=True)

    # Feature Scaling (normalizing the data)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Feature Engineering -  PCA for dimension reduction (Real coordinate space: metric/ binary)
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)



    # visualize imbalanced data
    plot_2d_space(X, y, 'Imbalanced dataset (2 PCA components)')

    # visualize data after oversampling
    X_ros, y_ros = implement_oversampling(X, X_train, y_train)
    plot_2d_space(X_ros, y_ros, 'Random over-sampling')
    return X_ros, y_ros, X_test, y_test


def classification(X_ros, y_ros, X_test, y_test):
    # Apply random forest as classification method
    rf_model = rf_classification(X_ros, y_ros)

    # Prediction and classification report including Accuracy and F1 score
    rf_report = pred_report(rf_model, X_test, y_test)
    return rf_report
