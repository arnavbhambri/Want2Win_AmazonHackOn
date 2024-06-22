import sqlite3
import pandas as pd

def setup_database(db_file):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    c.execute('''
              CREATE TABLE IF NOT EXISTS products (
                  Product_ID INTEGER PRIMARY KEY,
                  Product_Name TEXT NOT NULL,
                  Category TEXT,
                  Discounted_Price TEXT,
                  Actual_Price TEXT,
                  Discount_Percentage TEXT,
                  Rating TEXT,
                  Rating_Count TEXT,
                  About_Product TEXT,
                  Img_Link TEXT,
                  March INTEGER,
                  April INTEGER,
                  May INTEGER,
                  June INTEGER,
                  July INTEGER,
                  August INTEGER,
                  September INTEGER,
                  October INTEGER,
                  November INTEGER,
                  December INTEGER
              )
              ''')
    c.execute('''
              CREATE TABLE IF NOT EXISTS users (
                  user_id TEXT PRIMARY KEY,
                  password TEXT NOT NULL
              )
              ''')
    c.execute('''
              CREATE TABLE IF NOT EXISTS orders (
                  order_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  product_id INTEGER,
                  shipment_id TEXT,
                  FOREIGN KEY (user_id) REFERENCES users (user_id),
                  FOREIGN KEY (product_id) REFERENCES products (Product_ID)
              )
              ''')
    data = pd.read_csv(csv_file)
    for _, row in data.iterrows():
        c.execute('''
                  INSERT INTO products (
                      Product_ID, Product_Name, Category, Discounted_Price,
                      Actual_Price, Discount_Percentage, Rating, Rating_Count,
                      About_Product, Img_Link, March, April, May, June, July,
                      August, September, October, November, December
                  ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                  ''', (
                      row['Product_ID'], row['Product_Name'], row['Category'], row['Discounted_Price'],
                      row['Actual_Price'], row['Discount_Percentage'], row['Rating'], row['Rating_Count'],
                      row['About_Product'], row['Img_Link'], row['March'], row['April'], row['May'],
                      row['June'], row['July'], row['August'], row['September'], row['October'],
                      row['November'], row['December']
                  ))
    data2 = pd.read_csv("csv/user_database.csv")
    for _, row in data2.iterrows():
        c.execute('''
                  INSERT INTO users (user_id, password)
                  VALUES (?, ?)
                  ''', (row['User_ID'],row['Password']))
        
    data3 = pd.read_csv("csv/user_prod_map_with_shipment_id.csv")
    for _, row in data3.iterrows():
        c.execute('''
                  INSERT INTO orders (user_id, product_id,shipment_id)
                  VALUES (?,?,?)
                  ''', (row['user_id'], row['product_id'],row['Shipment_ID']))
    c.execute('''ALTER TABLE orders ADD COLUMN review_text TEXT''')

    # Add review_rating column
    c.execute('''ALTER TABLE orders ADD COLUMN review_rating INTEGER''')
    conn.commit()
    conn.close()

if __name__ == '__main__':
    csv_file = 'csv/product_database.csv'
    db_file = 'database.db'
    setup_database(db_file)
