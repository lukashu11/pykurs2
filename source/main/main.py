import pandas as pd
from source.dataflows.dataflow import feature_engineering, classification

#TODO: "src" nicht "source!"

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


def main():
    X_ros, y_ros, X_test, y_test = feature_engineering(olist_orders, olist_order_items, olist_products,
                                                       olist_order_payments, olist_order_reviews,
                                                       olist_customers)

    classification_report = classification(X_ros, y_ros, X_test, y_test)
    return classification_report

#TODO: keine prediction...

if __name__ == "__main__":
    print('Preparing Classification Report...')
    main()
