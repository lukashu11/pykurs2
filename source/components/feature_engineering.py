import pandas as pd


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


# left join calculated variables from olist_order_items and olist_products
def calc_count(df, groupkey, dict):
    df = df.groupby(groupkey).count().reset_index()
    df = df.rename(columns=dict)
    return df


def calc_mean(df, groupkey, dict):
    df = df.groupby(groupkey).mean().reset_index()
    df = df.rename(columns=dict)
    return df


def join_dfs(df_left, df_right, join_key, drop_nan):
    df = pd.merge(df_left, df_right, on=join_key, how='left')
    df = df.dropna(subset=drop_nan)
    return df


# def convert_datetime(df, column, drop_col):
# df['order_hour_minute_second'] = df[column].str.slice(start=11)
# df['order_hour_minute_second'] = pd.to_datetime(df[column], format='%H:%M:%S').dt.time
# df['order_year_month_day'] = pd.to_datetime(churn[column], format='%Y-%m-%d').dt.date
# df = df.drop(columns=drop_col).drop_duplicates()
# return df


from functools import reduce


def merge_calc_cols(df_to_merge, join_key):
    df = reduce(lambda left, right: pd.merge(left, right, on=join_key, how='left'), df_to_merge)
    df = df.dropna()
    return df


churn = olist_orders[olist_orders['order_status'].isin(['canceled', 'delivered'])]
churn = churn.drop(columns=['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date',
                            'order_estimated_delivery_date'])
olist_order_items = join_dfs(olist_order_items, olist_products, join_key=['product_id'], drop_nan=['product_id'])
count_order_items = calc_count(olist_order_items[['order_id', 'product_id']], groupkey=['order_id'],
                               dict={'product_id': 'count_order_items'})
avg_calc_order_items = calc_mean(olist_order_items[[
    'order_id', 'price', 'freight_value', 'product_name_lenght', 'product_description_lenght', 'product_name_lenght',
    'product_photos_qty',
    'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']],
                                 groupkey=['order_id'], dict={
        'price': 'avg_price',
        'freight_value': 'avg_freight_value',
        'product_name_lenght': 'avg_product_name_length',
        'product_description_lenght': 'avg_product_description_length',
        'product_name_lenght': 'avg_product_name_length',
        'product_photos_qty': 'avg_product_photos_qty',
        'product_weight_g': 'avg_product_weight_g',
        'product_length_cm': 'avg_product_length_cm',
        'product_height_cm': 'avg_product_height_cm',
        'product_width_cm': 'avg_product_width_cm'})

# avg_freight_value = calc_mean(olist_order_items[['order_id', 'freight_value']], groupkey=['order_id'], dict={'freight_value': 'avg_freight_value'})
# avg_product_name_length = calc_mean(olist_order_items[['order_id', 'product_name_lenght']], groupkey=['order_id'], dict={'product_name_lenght': 'avg_product_name_length'})
# avg_product_description_length = calc_mean(olist_order_items[['order_id', 'product_description_lenght']], groupkey=['order_id'], dict={'product_description_lenght': 'avg_product_description_length'})
# avg_product_photos_qty = calc_mean(olist_order_items[['order_id', 'product_name_lenght']], groupkey=['order_id'], dict={'product_name_lenght': 'avg_product_name_length'})
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

churn = join_dfs(churn, olist_customers[['customer_id', 'customer_state', 'customer_city']], join_key=['customer_id'],
                 drop_nan=['customer_id'])
churn = churn.set_index('order_id').dropna().drop(
    columns=['order_purchase_timestamp', 'customer_id', 'customer_city']).drop_duplicates()

churn['order_status'] = churn['order_status'].astype('category')
churn['customer_state'] = churn['customer_state'].astype('category')
churn['payment_type'] = churn['payment_type'].astype('category')

churn = pd.get_dummies(data=churn, columns=['order_status', 'customer_state', 'payment_type']).drop(columns=['order_status_delivered'])


# olist_order_items = pd.merge(olist_order_items, olist_products, on='product_id', how='left')
# count_order_items = olist_order_items[['order_id', 'product_id']].groupby(['order_id']).count().reset_index()
# count_order_items = count_order_items.rename(columns={'product_id': 'count_order_items'})
# churn = pd.merge(churn, count_order_items, on='order_id', how='left')

# avg_price = olist_order_items[['order_id', 'price']].groupby(['order_id']).mean().reset_index()
# avg_price = avg_price.rename(columns={'price': 'avg_price'})
# churn = pd.merge(churn, avg_price, on='order_id', how='left')

# avg_freight_value = olist_order_items[['order_id', 'freight_value']].groupby(['order_id']).mean().reset_index()
# avg_freight_value = avg_freight_value.rename(columns={'freight_value': 'avg_freight_value'})
# churn = pd.merge(churn, avg_freight_value, on='order_id', how='left')

# avg_product_name_length = olist_order_items[['order_id', 'product_name_lenght']].groupby(
#    ['order_id']).mean().reset_index()
# avg_product_name_length = avg_product_name_length.rename(columns={'product_name_lenght': 'avg_product_name_length'})
# churn = pd.merge(churn, avg_product_name_length, on='order_id', how='left')

# avg_product_description_lenght = olist_order_items[['order_id', 'product_description_lenght']].groupby(
#    ['order_id']).mean().reset_index()
# avg_product_description_lenght = avg_product_description_lenght.rename(
#    columns={'product_description_lenght': 'avg_product_description_length'})
# churn = pd.merge(churn, avg_product_description_lenght, on='order_id', how='left')

# avg_product_photos_qty = olist_order_items[['order_id', 'product_photos_qty']].groupby(
#   ['order_id']).mean().reset_index()
# avg_product_photos_qty = avg_product_photos_qty.rename(columns={'product_photos_qty': 'avg_product_photos_qty'})
# churn = pd.merge(churn, avg_product_photos_qty, on='order_id', how='left')

# avg_product_weight_g = olist_order_items[['order_id', 'product_weight_g']].groupby(['order_id']).mean().reset_index()
# avg_product_weight_g = avg_product_weight_g.rename(columns={'product_weight_g': 'avg_product_weight_g'})
# churn = pd.merge(churn, avg_product_weight_g, on='order_id', how='left')

# avg_product_length_cm = olist_order_items[['order_id', 'product_length_cm']].groupby(['order_id']).mean().reset_index()
# avg_product_length_cm = avg_product_length_cm.rename(columns={'product_length_cm': 'avg_product_length_cm'})
# churn = pd.merge(churn, avg_product_length_cm, on='order_id', how='left')

# avg_product_height_cm = olist_order_items[['order_id', 'product_height_cm']].groupby(['order_id']).mean().reset_index()
# avg_product_height_cm = avg_product_height_cm.rename(columns={'product_height_cm': 'avg_product_height_cm'})
# churn = pd.merge(churn, avg_product_height_cm, on='order_id', how='left')

# avg_product_width_cm = olist_order_items[['order_id', 'product_width_cm']].groupby(['order_id']).mean().reset_index()
# avg_product_width_cm = avg_product_width_cm.rename(columns={'product_width_cm': 'avg_product_width_cm'})
# churn = pd.merge(churn, avg_product_width_cm, on='order_id', how='left')

# left join varaibles needed from olist customers
# churn = pd.merge(churn, olist_customers[['customer_id', 'customer_state', 'customer_city']], on='customer_id',
#                 how='left')

# left join variables needed from olist_order_payments
# churn = pd.merge(churn, olist_order_payments[['order_id', 'payment_type']], on='order_id', how='left')

# left join calculated variables from olist_order_reviews
# count_product_comments = olist_order_reviews[['order_id', 'review_comment_message']].groupby(
#    ['order_id']).count().reset_index()
# count_product_comments = count_product_comments.rename(columns={'review_comment_message': 'count_product_comments'})
# churn = pd.merge(churn, count_product_comments, on='order_id', how='left')

# count_product_ratings = olist_order_reviews[['order_id', 'review_score']].groupby(['order_id']).count().reset_index()
# count_product_ratings = count_product_ratings.rename(columns={'review_score': 'count_product_ratings'})
# churn = pd.merge(churn, count_product_ratings, on='order_id', how='left')

# avg_rating = olist_order_reviews[['order_id', 'review_score']].groupby(['order_id']).mean().reset_index()
# avg_rating = avg_rating.rename(columns={'review_score': 'avg_rating'})
# churn = pd.merge(churn, avg_rating, on='order_id', how='left')

# set order_id as index

# convert to datetime
# churn['order_hour_minute_second'] = churn['order_purchase_timestamp'].str.slice(start=11)
# churn['order_hour_minute_second'] = pd.to_datetime(churn['order_hour_minute_second'], format='%H:%M:%S').dt.time
# churn['order_year_month_day'] = pd.to_datetime(churn['order_purchase_timestamp'], format='%Y-%m-%d').dt.date
# churn = churn

# churn = churn.drop_duplicates()

# get dummies of target variable other categorical variables (customer_state and payment_type)
#def get_categorical_dummies(df, drop_col):
    #dummies = pd.get_dummies(data=df, columns=['order_status', 'customer_state', 'payment_type'])
    #df = pd.concat([df, dummies], axis=1)
    #df = df.drop(columns=drop_col)
    #dummy_cols = ['order_status', 'customer_state', 'payment_type']
    #for column in dummy_cols:
        #dummies = df[column].str.get_dummies()
        #dummies.columns = ['is_' + col for col in df.columns]
        #df = pd.concat([df, dummies], axis=1)
        #df = df.drop(columns=drop_col)
    #return df


#churn = get_categorical_dummies(churn, drop_col=['order_status', 'is_delivered', 'customer_state', 'payment_type'])


#churn_status_dummies = churn.order_status.str.get_dummies()
#churn_status_dummies.columns = ['is_' + col for col in churn_status_dummies.columns]
#churn = pd.concat([churn, churn_status_dummies], axis=1)
#churn = churn.drop(columns=['order_status', 'is_delivered'])

#churn_state_dummies = churn.customer_state.str.get_dummies()
#churn_state_dummies.columns = ['is_' + col for col in churn_state_dummies.columns]
#churn = pd.concat([churn, churn_state_dummies], axis=1)
#churn = churn.drop(columns='customer_state')

#churn_payment_dummies = churn.payment_type.str.get_dummies()
#churn_payment_dummies.columns = ['is_' + col for col in churn_payment_dummies.columns]
#churn = pd.concat([churn, churn_payment_dummies], axis=1)
#churn = churn.drop(columns='payment_type')
