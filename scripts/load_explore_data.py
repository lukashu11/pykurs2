import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# load different datasets
olist_orders = pd.read_csv(
    r'C:\Users\lhube\OneDrive - hs-pforzheim.de\HS Pforzheim\WS2021\Python Kurs II\archive\olist_orders_dataset.csv')
olist_order_items = pd.read_csv(
    r'C:\Users\lhube\OneDrive - hs-pforzheim.de\HS Pforzheim\WS2021\Python Kurs II\archive\olist_order_items_dataset.csv')
olist_order_reviews = pd.read_csv(
    r'C:\Users\lhube\OneDrive - hs-pforzheim.de\HS Pforzheim\WS2021\Python Kurs II\archive\olist_order_reviews_dataset.csv')
olist_order_payments = pd.read_csv(
    r'C:\Users\lhube\OneDrive - hs-pforzheim.de\HS Pforzheim\WS2021\Python Kurs II\archive\olist_order_payments_dataset.csv')
olist_customers = pd.read_csv(
    r'C:\Users\lhube\OneDrive - hs-pforzheim.de\HS Pforzheim\WS2021\Python Kurs II\archive\olist_customers_dataset.csv')
olist_geolocation = pd.read_csv(
    r'C:\Users\lhube\OneDrive - hs-pforzheim.de\HS Pforzheim\WS2021\Python Kurs II\archive\olist_geolocation_dataset.csv')
# olist_sellers = pd.read_csv(
    # r'C:\Users\lhube\OneDrive - hs-pforzheim.de\HS Pforzheim\WS2021\Python Kurs II\archive\olist_sellers_dataset.csv')
olist_products = pd.read_csv(
    r'C:\Users\lhube\OneDrive - hs-pforzheim.de\HS Pforzheim\WS2021\Python Kurs II\archive\olist_products_dataset.csv')

# explore olist orders and order items
# print(olist_orders.describe())

# test_orders = olist_orders.head(10)

# olist_orders['order_delivered_customer_date'].isnull().sum()

# print(olist_order_items.describe())

# test_items = olist_order_items.head(10)

# olist_order_items['price'].isnull().sum()

# test_customers = olist_customers.head(10)

# test_geolocation = olist_geolocation.head(10)

# test_sellers = olist_sellers.head(10)

# test_products = olist_products.head(10)

# print(olist_orders['order_status'].describe())
# olist_orders.order_status.unique()

count_status = olist_orders[['order_id', 'order_status']].groupby(['order_status']).count().reset_index()

count_status = count_status[count_status['order_status'].isin(['canceled', 'delivered'])]

plt.figure(figsize=(16, 8))
# plot chart
ax1 = plt.subplot(121, aspect='equal')
count_status.plot(kind='pie', y='order_id', ax=ax1, autopct='%1.1f%%',
                  startangle=90, shadow=False, labels=count_status['order_status'], legend=True, fontsize=14)

# convert to datetime format


# convert to boolean (hasdate)
# date1 = pd.to_datetime(olist_orders['order_purchase_timestamp'], errors='coerce').notna()
# date2 = olist_orders['order_purchase_timestamp'].notna()
# olist_orders['order_purchase_timestamp'] = np.select([date1, date2], [True, False], None)

# date3 = pd.to_datetime(olist_orders['order_approved_at'], errors='coerce').notna()
# date4 = olist_orders['order_approved_at'].notna()
# olist_orders['order_approved_at'] = np.select([date3, date4], [True, False], None)

# date5 = pd.to_datetime(olist_orders['order_delivered_carrier_date'], errors='coerce').notna()
# date6 = olist_orders['order_delivered_carrier_date'].notna()
# olist_orders['order_delivered_carrier_date'] = np.select([date5, date6], [True, False], None)

# date7 = pd.to_datetime(olist_orders['order_delivered_customer_date'], errors='coerce').notna()
# date8 = olist_orders['order_delivered_customer_date'].notna()
# olist_orders['order_delivered_customer_date'] = np.select([date7, date8], [True, False], None)


# groupby order id (aggregation on order level --> space transformation of product level to order level via count/ mean etc.) to count features/ get mean, join dataframes and drop not needed columns

churn = olist_orders[olist_orders['order_status'].isin(['canceled', 'delivered'])]
churn = churn.drop(columns=['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date',
                            'order_estimated_delivery_date'])

olist_order_items = pd.merge(olist_order_items, olist_products, on='product_id', how='left')

count_order_items = olist_order_items[['order_id', 'product_id']].groupby(['order_id']).count().reset_index()
count_order_items = count_order_items.rename(columns={'product_id': 'count_order_items'})
churn = pd.merge(churn, count_order_items, on='order_id', how='left')

avg_price = olist_order_items[['order_id', 'price']].groupby(['order_id']).mean().reset_index()
avg_price = avg_price.rename(columns={'price': 'avg_price'})
churn = pd.merge(churn, avg_price, on='order_id', how='left')

avg_freight_value = olist_order_items[['order_id', 'freight_value']].groupby(['order_id']).mean().reset_index()
avg_freight_value = avg_freight_value.rename(columns={'freight_value': 'avg_freight_value'})
churn = pd.merge(churn, avg_freight_value, on='order_id', how='left')

avg_product_name_length = olist_order_items[['order_id', 'product_name_lenght']].groupby(
    ['order_id']).mean().reset_index()
avg_product_name_length = avg_product_name_length.rename(columns={'product_name_lenght': 'avg_product_name_length'})
churn = pd.merge(churn, avg_product_name_length, on='order_id', how='left')

avg_product_description_lenght = olist_order_items[['order_id', 'product_description_lenght']].groupby(
    ['order_id']).mean().reset_index()
avg_product_description_lenght = avg_product_description_lenght.rename(
    columns={'product_description_lenght': 'avg_product_description_length'})
churn = pd.merge(churn, avg_product_description_lenght, on='order_id', how='left')

avg_product_photos_qty = olist_order_items[['order_id', 'product_photos_qty']].groupby(
    ['order_id']).mean().reset_index()
avg_product_photos_qty = avg_product_photos_qty.rename(columns={'product_photos_qty': 'avg_product_photos_qty'})
churn = pd.merge(churn, avg_product_photos_qty, on='order_id', how='left')

avg_product_weight_g = olist_order_items[['order_id', 'product_weight_g']].groupby(['order_id']).mean().reset_index()
avg_product_weight_g = avg_product_weight_g.rename(columns={'product_weight_g': 'avg_product_weight_g'})
churn = pd.merge(churn, avg_product_weight_g, on='order_id', how='left')

avg_product_length_cm = olist_order_items[['order_id', 'product_length_cm']].groupby(['order_id']).mean().reset_index()
avg_product_length_cm = avg_product_length_cm.rename(columns={'product_length_cm': 'avg_product_length_cm'})
churn = pd.merge(churn, avg_product_length_cm, on='order_id', how='left')

avg_product_height_cm = olist_order_items[['order_id', 'product_height_cm']].groupby(['order_id']).mean().reset_index()
avg_product_height_cm = avg_product_height_cm.rename(columns={'product_height_cm': 'avg_product_height_cm'})
churn = pd.merge(churn, avg_product_height_cm, on='order_id', how='left')

avg_product_width_cm = olist_order_items[['order_id', 'product_width_cm']].groupby(['order_id']).mean().reset_index()
avg_product_width_cm = avg_product_width_cm.rename(columns={'product_width_cm': 'avg_product_width_cm'})
churn = pd.merge(churn, avg_product_width_cm, on='order_id', how='left')

# churn = pd.merge(churn, olist_order_reviews[['order_id', 'review_score', 'review_comment_message']], on = 'order_id', how = 'left')

churn = pd.merge(churn, olist_customers[['customer_id', 'customer_state', 'customer_city']], on='customer_id',
                 how='left')

churn = pd.merge(churn, olist_order_payments[['order_id', 'payment_type']], on='order_id', how='left')

# churn = pd.merge(churn, olist_sellers[['seller_id', 'seller_state']], on= 'seller_id', how = 'left')

count_product_comments = olist_order_reviews[['order_id', 'review_comment_message']].groupby(
    ['order_id']).count().reset_index()
count_product_comments = count_product_comments.rename(columns={'review_comment_message': 'count_product_comments'})
churn = pd.merge(churn, count_product_comments, on='order_id', how='left')

count_product_ratings = olist_order_reviews[['order_id', 'review_score']].groupby(['order_id']).count().reset_index()
count_product_ratings = count_product_ratings.rename(columns={'review_score': 'count_product_ratings'})
churn = pd.merge(churn, count_product_ratings, on='order_id', how='left')

avg_rating = olist_order_reviews[['order_id', 'review_score']].groupby(['order_id']).mean().reset_index()
avg_rating = avg_rating.rename(columns={'review_score': 'avg_rating'})
churn = pd.merge(churn, avg_rating, on='order_id', how='left')

# check for patterns in cancellations
churn.groupby(["customer_state", "order_status"]).size().unstack().plot(kind='bar', stacked=True, figsize=(30, 10))
churn.groupby(["avg_product_photos_qty", "order_status"]).size().unstack().plot(kind='bar', stacked=True,
                                                                                figsize=(30, 10))
churn.groupby(["avg_rating", 'order_status']).size().unstack().plot(kind='bar', stacked=True, figsize=(30, 10))
churn.groupby(["payment_type", "order_status"]).size().unstack().plot(kind='bar', stacked=True, figsize=(30, 10))
churn.groupby(["count_order_items", "order_status"]).size().unstack().plot(kind='bar', stacked=True, figsize=(30, 10))

churn = churn.set_index('order_id')

# drop NAs
churn = churn.dropna()

# convert to datetime
churn['order_hour_minute_second'] = churn['order_purchase_timestamp'].str.slice(start=11)
churn['order_hour_minute_second'] = pd.to_datetime(churn['order_hour_minute_second'], format='%H:%M:%S').dt.time
churn['order_year_month_day'] = pd.to_datetime(churn['order_purchase_timestamp'], format='%Y-%m-%d').dt.date
churn = churn.drop(columns= ['order_purchase_timestamp', 'customer_id', 'customer_city'])


churn = churn.drop_duplicates()

# get dummies of target variable other categorical variables (customer_state and payment_type)
churn['order_status'] = churn['order_status'].astype('category')
churn['customer_state'] = churn['customer_state'].astype('category')
churn['payment_type'] = churn['payment_type'].astype('category')

churn_status_dummies = churn.order_status.str.get_dummies()
churn_status_dummies.columns = ['is_' + col for col in churn_status_dummies.columns]
churn = pd.concat([churn, churn_status_dummies], axis = 1)
churn = churn.drop(columns= ['order_status', 'is_delivered'])

churn_state_dummies = churn.customer_state.str.get_dummies()
churn_state_dummies.columns = ['is_' + col for col in churn_state_dummies.columns]
churn = pd.concat([churn, churn_state_dummies], axis = 1)
churn = churn.drop(columns= 'customer_state')

churn_payment_dummies = churn.payment_type.str.get_dummies()
churn_payment_dummies.columns = ['is_' + col for col in churn_payment_dummies.columns]
churn = pd.concat([churn, churn_payment_dummies], axis = 1)
churn = churn.drop(columns= 'payment_type')

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



# order (1.Train Test Split, 2. Normalize data, 3. PCA, 4. Over/Undersampling, 5. Classification/ Prediction)?
# linear discriminant analysis valid? - PCA recommended for highly skewed data
# check space transformations (adapt DAG2 with PCA)





# TARGET predict order cancellations/ churns -> classification problem
# Entry data - different metric, nominal etc. features aggregated (mean, count etc.) on order level
# Output data - binary space (canceled or not canceled)
## orders, order items, products, reviews, customers, (geolocation), payments (multiple options?) needed
## possible important features: customer location (state/ city), number of complaints (comments),  average review score, payment method, product category, count comments, count reviews, product name/description length, price, freight value
### possible additional features:  (product category), product photos, product measures time-series data, (geolocation)




## join relevant tables via primary keys
## remove NAs or impute data for delivery date (carrier/customer)
## convert date columns to datetime format
## check for anomalies (null values/ negative values etc.)
## filter for status = cancelled/ delivered
## further analyse the data with plots and descriptive analysis (e.g. identify patterns etc.)





