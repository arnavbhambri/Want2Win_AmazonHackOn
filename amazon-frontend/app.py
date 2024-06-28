from flask import Flask, render_template, request, redirect, url_for, flash, session,jsonify
import sqlite3
import torch
import joblib
from werkzeug.security import generate_password_hash, check_password_hash
import os
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import networkx as nx
from pyvis.network import Network
import pandas as pd
import networkx as nx
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re
import torch
from create_graph import getnet
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
app.secret_key = 'skey'

MODEL_FILE = 'model/sbert_model.pkl'
EMBEDDINGS_FILE = 'model/product_embeddings.pkl'
df = pd.read_csv('csv/updated_product_database.csv')
with open(MODEL_FILE, 'rb') as f:
    title_model = pickle.load(f)
with open(EMBEDDINGS_FILE, 'rb') as f:
    product_embeddings = pickle.load(f)
# Load the description similarity model
model_save_path = "model/paraphrase-MiniLM-L6-v2-model.pth"
desc_model = torch.load(model_save_path,map_location=torch.device('cpu'))

@app.template_filter('truncate')
def truncate(s, length=200):
    if len(s) <= length:
        return s
    return s[:length] + '...'

# Register the filter
app.jinja_env.filters['truncate'] = truncate

loaded_model = torch.jit.load('model/model2_scripted.pth')

# Ensure you set the model to evaluation mode
loaded_model.eval()

# Load the saved vectorizer
vectorizer = joblib.load('model/vectorizer.pkl')
shipments_df = pd.read_csv('csv/shipment_database1.csv')
status_df = pd.read_csv('csv/shipment_status_database1.csv')
users_df = pd.read_csv('csv/user_database_normalized.csv')
hotspots_df = pd.read_csv('csv/hotspots.csv')

# Load your dataset

# Encode all existing descriptions
def get_seller_data():
    return pd.read_csv('csv/seller_auth.csv')

def get_product_data():
    return pd.read_csv('csv/seller_database.csv')

@app.route('/products3', methods=['GET', 'POST'])
def products3():
    if 'seller_id' not in session:
        flash('Please log in to view products')
        return redirect(url_for('login2'))

    seller_id = session['seller_id']
    product_data = get_product_data()

    if request.method == 'POST':
        search_value = request.form.get('search_value', '').lower()
        product_data = product_data[
            product_data['Product_Name'].str.lower().str.contains(search_value) |
            product_data['Category'].str.lower().str.contains(search_value) |
            product_data['About_Product'].str.lower().str.contains(search_value)
        ]

    return render_template('products3.html', products=product_data.to_dict('records'))

@app.route('/add_to_catalog/<int:product_id>', methods=['GET', 'POST'])
def add_to_catalog(product_id):
    if 'seller_id' not in session:
        flash('Please log in to add products to your catalog')
        return redirect(url_for('login2'))

    product_data = get_product_data()
    product = product_data[product_data['Product_ID'] == product_id].iloc[0]

    if request.method == 'POST':
        discounted_price = request.form['discounted_price']
        about_product = request.form['about_product']
        to=pd.read_csv('csv/seller_database.csv')
        we=to[to['Product_ID']==product_id]
        we['About_Product']=about_product
        we['Discounted_Price']=discounted_price
        seller_id = session['seller_id']
        we['Seller_ID']=seller_id
        we["Reg_ID"]=len(to)+1
        to = to._append(we,ignore_index=True)
        to.to_csv('csv/seller_database.csv')
        # Logic to add product to the seller's catalog with updated details
        flash(f'Product {product_id} added to your catalog with new details!')
        return redirect(url_for('products2'))

    return render_template('add_to_catalog.html', product=product)

@app.route('/seller_login')
def seller_login():
    return render_template('index2.html')

@app.route('/login2', methods=['GET', 'POST'])
def login2():
    if request.method == 'POST':
        seller_id = request.form['seller_id']
        password = request.form['password']
        
        seller_data = get_seller_data()
        seller = seller_data[seller_data['Seller_ID'] == seller_id]
        if not seller.empty and seller.iloc[0]['Password'] == password:
            session['seller_id'] = seller_id
            flash('Login successful')
            return redirect(url_for('products2'))
        else:
            flash('Invalid seller ID or password')
    return render_template('login2.html')

@app.route('/register2', methods=['GET', 'POST'])
def register2():
    if request.method == 'POST':
        seller_id = request.form['seller_id']
        password = request.form['password']
        
        seller_data = get_seller_data()
        if seller_id in seller_data['Seller_ID'].values:
            flash('Seller ID already exists')
        else:
            new_seller = pd.DataFrame({'Seller_ID': [seller_id], 'Password': [password]})
            seller_data = pd.concat([seller_data, new_seller], ignore_index=True)
            seller_data.to_csv('csv/seller_auth.csv', index=False)
            flash('Registration successful, please log in')
            return redirect(url_for('login2'))
    return render_template('register2.html')

@app.route('/logout')
def logout():
    session.pop('seller_id', None)
    flash('You have been logged out')
    return redirect(url_for('seller_login'))

@app.route('/products2', methods=['GET', 'POST'])
def products2():
    if 'seller_id' not in session:
        flash('Please log in to view products')
        return redirect(url_for('login2'))

    seller_id = session['seller_id']
    product_data = get_product_data()
    seller_products = product_data[product_data['Seller_ID'] == seller_id]

    if request.method == 'POST':
        search_value = request.form.get('search_value', '').lower()
        seller_products = seller_products[
            seller_products['Product_Name'].str.lower().str.contains(search_value) |
            seller_products['Category'].str.lower().str.contains(search_value) |
            seller_products['About_Product'].str.lower().str.contains(search_value)
        ]

    return render_template('products2.html', products=seller_products.to_dict('records'))

@app.route('/product_registration', methods=['GET', 'POST'])
def product_registration():
    if 'seller_id' not in session:
        flash('Please log in to register a product')
        return redirect(url_for('login2'))

    if request.method == 'POST':
        seller_id = session['seller_id']
        product_id = request.form['product_id']
        product_name = request.form['product_name']
        category = request.form['category']
        discounted_price = request.form['discounted_price']
        actual_price = request.form['actual_price']
        about_product = request.form['about_product']
        img_link = request.form['img_link']

        product_data = get_product_data()
        # Check if Product_ID exists in seller_database.csv
        if int(product_id) in product_data['Product_ID'].values:
            flash('Product ID already exists')
            return render_template('product_registration.html')

        try:
            # Append to CSV file
            new_product = pd.DataFrame({
                'Product_ID': [product_id],
                'Seller_ID': [seller_id],
                'Product_Name': [product_name],
                'Category': [category],
                'Discounted_Price': [discounted_price],
                'Actual_Price': [actual_price],
                'About_Product': [about_product],
                'Img_Link': [img_link],
            })
            
            product_data = pd.concat([product_data, new_product], ignore_index=True)
            product_data.to_csv("csv/seller_database.csv", index=False)
            
            flash('Product registered successfully')
            return redirect(url_for('products2'))
        except Exception as e:
            print(f"Error: {e}")
            flash('An error occurred during product registration. Please try again.')

    return render_template('product_registration.html')

@app.route('/check_similarity', methods=['POST'])
def check_similarity_route():
    descriptions = df['About_Product'].tolist()
    new_description = request.form['about_product']
    dimensions = request.form['dimensions']
    product_name = request.form['product_name']
    def find_similar_titles(input_title, top_n=50):
        input_embedding = title_model.encode([input_title])
        similarities = cosine_similarity(input_embedding, product_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        return df.iloc[top_indices]
    def parse_dimensions(dimension_string):
        match = re.match(r'(\d+)x(\d+)x(\d+)', dimension_string)
        return tuple(map(int, match.groups())) if match else None

    def similarity(dim1, dim2, threshold=2):
        return all(abs(d1 - d2) <= threshold for d1, d2 in zip(dim1, dim2))

    def find_similar_products(products, target_dims, threshold=2):
        similar_products = []
        for _, product in products.iterrows():
            dims = parse_dimensions(product['Dimensions'])
            if dims and similarity(dims, target_dims, threshold):
                similar_products.append(product)
        return pd.DataFrame(similar_products)
    def calculate_similarity(new_description, embeddings, model2):
        new_embedding = model2.encode(new_description, convert_to_tensor=True)
        similarity_scores = util.pytorch_cos_sim(new_embedding, embeddings)
        return similarity_scores
    similar_titles = find_similar_titles(product_name)
    target_dims = parse_dimensions(dimensions)
    if not target_dims:
        print("Invalid dimension format. Please use the format LxWxH (e.g., 10x5x2).")
    similar_products = find_similar_products(similar_titles, target_dims)
    embeddings2 = desc_model.encode(similar_products['About_Product'].tolist(), convert_to_tensor=True)
    similarity_scores = calculate_similarity(new_description, embeddings2, desc_model)
    is_too_similar = (similarity_scores > 0.7).any().item()

    result = f"Is the new description too similar? {'Yes' if is_too_similar else 'No'}\n"

    most_similar_index = similarity_scores.argmax().item()
    most_similar_score = similarity_scores[0, most_similar_index].item()
    simp=similar_products['About_Product'].tolist()
    if most_similar_index < len(similar_products):
        most_similar_description = simp[most_similar_index]
        result += f"Most similar description: {most_similar_description}\n"
        result += f"Similarity score with the most similar description: {most_similar_score:.2f}"
    else:
        result += "Error: Most similar index is out of range."
    return jsonify({'result': result, 'is_too_similar': is_too_similar})

def predict_statement(statement, vectorizer):
    # Preprocess the input statement
    statement_vectorized = vectorizer.transform([statement]).toarray()
    statement_tensor = torch.tensor(statement_vectorized, dtype=torch.float32)
    
    # Make a prediction
    with torch.no_grad():
        output = loaded_model(statement_tensor)
        predicted_class = (output > 0.5).float().item()  # Threshold at 0.5 for binary classification
        return int(predicted_class)  # Convert to integer for readability (0 or 1)
    
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/review/<int:product_id>', methods=['GET', 'POST'])
def review_product(product_id):
    if 'user_id' not in session:
        flash('Please log in to submit a review')
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = get_db_connection()

    product = conn.execute('SELECT * FROM products WHERE Product_ID = ?', (product_id,)).fetchone()
    existing_review = conn.execute('''
        SELECT review_rating, review_text
        FROM orders
        WHERE user_id = ? AND product_id = ?
    ''', (user_id, product_id)).fetchone()

    conn.close()

    if not product:
        flash('Product not found')
        return redirect(url_for('products'))
    return render_template('review.html', product=product, existing_review=existing_review)


@app.route('/submit_review/<int:product_id>', methods=['POST'])
def submit_review(product_id):
    if 'user_id' not in session:
        flash('Please log in to submit a review')
        return redirect(url_for('login'))

    user_id = session['user_id']
    review_rating = request.form['review_rating']
    review_text = request.form['review_text']

    # Predict the class of the review text
    predicted_class = predict_statement(review_text, vectorizer)
    
    if predicted_class == 1:
        flash('Your review was detected as spam and cannot be submitted.')
        return redirect(url_for('review_product', product_id=product_id))
    
    # If the review is genuine, proceed to save it
    conn = get_db_connection()
    conn.execute('''
        UPDATE orders
        SET review_rating = ?, review_text = ?
        WHERE user_id = ? AND product_id = ?
    ''', (review_rating, review_text, user_id, product_id))
    conn.commit()
    conn.close()

    flash('Review submitted successfully')
    return redirect(url_for('products'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['user_id']
        password = request.form['password']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone()
        conn.close()
        if user and user['password']==password:
            session['user_id'] = user['user_id']
            return redirect(url_for('products'))
        else:
            flash('Invalid user ID or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_id = request.form['user_id']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='sha256')
        conn = get_db_connection()
        conn.execute('INSERT INTO users (user_id, password) VALUES (?, ?)', (user_id, password))
        conn.commit()
        conn.close()
        flash('Registration successful, please log in')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/products', methods=['GET', 'POST'])
def products():
    if 'user_id' not in session:
        flash('Please log in to view products')
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = get_db_connection()

    if request.method == 'POST':
        search_value = request.form.get('search_value', '')
        search_value = f"%{search_value}%"
        products = conn.execute('''
            SELECT * FROM products 
            WHERE Product_Name LIKE ? 
            OR Category LIKE ? 
            OR About_Product LIKE ?
        ''', (search_value, search_value, search_value)).fetchall()
    else:
        products = conn.execute('''
            SELECT p.*
            FROM products p
            JOIN orders o ON p.Product_ID = o.product_id
            WHERE o.user_id = ?
        ''', (user_id,)).fetchall()
        
    conn.close()
    return render_template('products.html', products=products)

@app.route('/return/<int:product_id>', methods=['GET', 'POST'])
def return_form(product_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    conn = get_db_connection()
    
    product = conn.execute('SELECT * FROM products WHERE Product_ID = ?', (product_id,)).fetchone()
    shipment = conn.execute('SELECT * FROM orders WHERE Product_ID = ? AND USER_ID = ?', (product_id, user_id)).fetchone()
    conn.close()

    if not product:
        flash('Product not found')
        return redirect(url_for('products'))

    if request.method == 'POST':
        reason = request.form['reason']
        counterfeit = 'counterfeit' in request.form
        # Perform your function here using user_id, shipment['Shipment_ID'], reason, and counterfeit
        process_return(user_id, shipment['Shipment_ID'],product['Product_ID'], reason, counterfeit)
        flash('Return processed successfully.')
        return redirect(url_for('products'))

    return render_template('return_form.html', product=product, user_id=user_id, shipment=shipment)

def process_return(user_id, shipment_id,product_id, reason, counterfeit):
    # Implement your function logic here
    # For example, update the database with the return details

    # Check if user exists in the database
    if user_id not in users_df['User_ID'].astype(str).values:
        return "Invalid user"

        # Check if shipment_id and product_id correspond
    matching_orders = shipments_df[(shipments_df['Shipment_ID'].astype(str) == shipment_id)]
        
    if matching_orders.empty:
        print(f"{product_id}")
        return f"Invalid order entered. Shipment ID: {shipment_id}, Product ID: {product_id}"

        # If we've reached this point, the user and order are valid
        # Get counterfeit score
    counterfeit_score = status_df[status_df['Shipment_ID'].astype(str) == shipment_id]['Counterfeit_Score'].values[0]

        # Get user legitimacy score
    user_legitimacy = users_df[users_df['User_ID'].astype(str) == user_id]['Normalized_Legitimacy_Score'].values[0]

        # Get warehouse ID from shipment
    warehouse_id = matching_orders['Warehouse_ID'].values[0]

        # Check if warehouse is in hotspots and calculate average weight
    hotspot_weights = hotspots_df[
        (hotspots_df['H1'].astype(str) == str(warehouse_id)) | 
        (hotspots_df['H2'].astype(str) == str(warehouse_id))
    ]['Weight'].values

    hot=hotspots_df['Weight'].values
    hotspot_weight = hotspot_weights.mean() if len(hotspot_weights) > 0 else 0
    hotspot_weightm = max(hot)
    hotspot_weightma = min(hot)
    htp = (hotspot_weight - hotspot_weightma) / (hotspot_weightm - hotspot_weightma)

        # Determine counterfeit status
    sas=1-user_legitimacy
    if counterfeit and user_legitimacy>=0.8:
        status=1
    elif counterfeit and user_legitimacy<=0.2:
        status=0.1
    elif counterfeit:
        status=user_legitimacy*0.666+htp*0.333
    else:
        status=user_legitimacy*0.20+ htp*0.40
        # Add to return_complaints.csv
    status_df.loc[status_df['Shipment_ID'].astype(str) == shipment_id, 'Counterfeit_Score'] = status 
    status_df.loc[status_df['Shipment_ID'].astype(str) == shipment_id, 'Return_Initiated'] = True
    status_df.to_csv('csv/shipment_status_database1.csv')
    return "Report submitted successfully"

@app.route('/generate_graph')
def generate_graph():
    #Code for reporting first
    #products
    im=pd.read_csv('csv/shipment_status_database1.csv')
    ship=pd.read_csv('csv/shipment_database1.csv')
    to=pd.read_csv('csv/seller_database.csv')
    prod=pd.read_csv('csv/product_database.csv')
    sh2=ship[['Shipment_ID','Seller_ID']]
    sh3=ship[['Shipment_ID','Product_ID']]
    pdt=pd.merge(im,sh3,how='inner',on='Shipment_ID')
    cnt={}
    for ind in pdt.index:
        if pdt['Counterfeit_Score'][ind]>0.70:
            seller=pdt['Product_ID'][ind]
            if seller not in cnt:
                cnt[seller]=1
            else:
                cnt[seller]+=1
    for i in cnt:
        lent=pdt['Product_ID'].value_counts().get(i, 0)
        nt=cnt[i]/lent
        y=round(nt*100,2)
        cnt[i]=y
    keys = list(cnt.keys())
    values = list(cnt.values())
    sorted_value_index = np.argsort(values)
    sorted_value_index=np.flip(sorted_value_index)
    sorted_dict = {keys[i]: values[i] for i in sorted_value_index}
    product_percentages = dict(list(sorted_dict.items())[:10])
    #sellers
    pdx=pd.merge(im,sh2,how='inner',on='Shipment_ID')
    cnt={}
    for ind in pdx.index:
        if pdx['Counterfeit_Score'][ind]>0.70:
            seller=pdx['Seller_ID'][ind]
            if seller not in cnt:
                cnt[seller]=1
            else:
                cnt[seller]+=1
    for i in cnt:
        lent=pdx['Seller_ID'].value_counts().get(i, 0)
        nt=cnt[i]/lent
        y=round(nt*100,2)
        cnt[i]=y
    keys = list(cnt.keys())
    values = list(cnt.values())
    sorted_value_index = np.argsort(values)
    sorted_value_index=np.flip(sorted_value_index)
    sorted_dict = {keys[i]: values[i] for i in sorted_value_index}
    seller_percentages = dict(list(sorted_dict.items())[:10])
    #Code for graph- refer create_graph.py
    net = getnet()
    graph_html = net.generate_html()

    graph_with_title = f"""
    <h2 style="text-align: center;">Counterfeiting Hotspots</h2>
    {graph_html}
    """
    return render_template('show_graph.html', graph_html=graph_with_title, seller_percentages=seller_percentages, product_percentages=product_percentages)


if __name__ == '__main__':
    app.run(debug=True)
