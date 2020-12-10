import pandas as pd
from src.components.feature_engineering import calc_mean, calc_count, join_dfs, merge_calc_cols, split_data, \
    plot_2d_space, implement_oversampling, scale_features, reduce_dimensions

# load different datasets needed
olist_orders = pd.read_csv(
    './data/olist_orders_dataset.csv', encoding='utf-8')
olist_order_items = pd.read_csv(
    './data/olist_order_items_dataset.csv', encoding='utf-8')
olist_order_reviews = pd.read_csv(
    './data/olist_order_reviews_dataset.csv', encoding='utf-8')
olist_order_payments = pd.read_csv(
    './data/olist_order_payments_dataset.csv', encoding='utf-8')
olist_customers = pd.read_csv(
    './data/olist_customers_dataset.csv', encoding='utf-8')
olist_geolocation = pd.read_csv(
    './data/olist_geolocation_dataset.csv', encoding='utf-8')
olist_products = pd.read_csv(
    './data/olist_products_dataset.csv', encoding='utf-8')

# get target variable and drop not needed columns from olist_orders
churn = olist_orders[olist_orders['order_status'].isin(['canceled', 'delivered'])]
churn = churn.drop(columns=['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date',
                            'order_estimated_delivery_date'])

# join in ald calculate needed columns from different dataframes
olist_order_items = join_dfs(olist_order_items, olist_products, join_key=['product_id'], drop_nan=['product_id'])
count_order_items = calc_count(olist_order_items[['order_id', 'product_id']], groupkey=['order_id'])
avg_calc_order_items = calc_mean(olist_order_items[[
    'order_id', 'price', 'freight_value', 'product_name_lenght', 'product_description_lenght',
    'product_photos_qty',
    'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']],
                                 groupkey=['order_id'])

olist_order_payments_reduced = olist_order_payments[['order_id', 'payment_type']]

count_calc_reviews = calc_count(olist_order_reviews[['order_id', 'review_comment_message', 'review_score']],
                                groupkey='order_id')

avg_calc_ratings = calc_mean(olist_order_reviews[['order_id', 'review_score']], groupkey='order_id')

df_to_merge = [churn, count_order_items, avg_calc_order_items, olist_order_payments_reduced, count_calc_reviews,
               avg_calc_ratings]
churn = merge_calc_cols(df_to_merge, join_key=['order_id'])

churn = join_dfs(churn, olist_customers[['customer_id', 'customer_state', 'customer_city']], join_key=['customer_id'],
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
