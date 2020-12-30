import pandas as pd
import numpy as np
from src.project.dataflow import flow_new_data, flow_train_data

#TODO: Hinzugefügt für bessere Anzeige in Python Console
pd.set_option('display.width', 700)
pd.options.display.max_colwidth = 100
np.set_printoptions(linewidth=800)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)

# TODO: Sie haben nirgends numpydoc verwendet (oder ähnliches)!
# TODO: Sie haben kein typgin verwendet!

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
    rf_report = flow_train_data(olist_orders, olist_order_items, olist_products,
                                                       olist_order_payments, olist_order_reviews,
                                                       olist_customers, path='./data/rf_model.pkl')
    new_data = flow_new_data(olist_orders, olist_order_items, olist_products,
                                                       olist_order_payments, olist_order_reviews,
                                                       olist_customers, model_path='./data/rf_model.pkl')

    return rf_report, new_data


if __name__ == "__main__":
    print('Preparing Classification Report and Prediction of new input data...')
    main()
