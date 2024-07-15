import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense
import os

# Load dataset
data = pd.read_csv('myntra_products.csv')

# Handle missing values
data['DiscountPrice (in Rs)'] = data['DiscountPrice (in Rs)'].replace('', np.nan).astype(float).fillna(0)
data['OriginalPrice (in Rs)'] = data['OriginalPrice (in Rs)'].replace('', np.nan).astype(float).fillna(0)
data['Ratings'] = data['Ratings'].replace('', np.nan).astype(float).fillna(0)
data['Reviews'] = data['Reviews'].replace('', np.nan).astype(float).fillna(0).astype(int)

# Convert categorical data to numerical using Label Encoding
label_encoders = {}
categorical_columns = ['BrandName', 'Category', 'Individual_category', 'category_by_Gender', 'SizeOption']

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Normalize numerical data
scaler = StandardScaler()
data[['DiscountPrice (in Rs)', 'OriginalPrice (in Rs)', 'Ratings', 'Reviews']] = scaler.fit_transform(data[['DiscountPrice (in Rs)', 'OriginalPrice (in Rs)', 'Ratings', 'Reviews']])

# Collaborative Filtering Model
n_users = len(data['Product_id'].unique())
n_items = len(data['Product_id'].unique())

user_input = Input(shape=(1,))
item_input = Input(shape=(1,))

user_embedding = Embedding(input_dim=n_users, output_dim=50)(user_input)
item_embedding = Embedding(input_dim=n_items, output_dim=50)(item_input)

user_vecs = Flatten()(user_embedding)
item_vecs = Flatten()(item_embedding)

y = Dot(axes=1)([user_vecs, item_vecs])

model = Model([user_input, item_input], y)
model.compile(optimizer='adam', loss='mean_squared_error')

# Dummy training data
user_ids = np.random.randint(0, n_users, 10000)
item_ids = np.random.randint(0, n_items, 10000)
ratings = np.random.uniform(0, 5, 10000)

model_path = 'collab_filter_model.h5'

if not os.path.exists(model_path):
    # Train the model if it doesn't exist
    model.fit([user_ids, item_ids], ratings, epochs=5, verbose=1)
    model.save(model_path)
else:
    # Load the model if it exists
    model = load_model(model_path)

# Content-Based Filtering Model
data_subset = data.head(100)  # Use a smaller subset for testing
item_features = data_subset[['BrandName', 'Category', 'Individual_category', 'category_by_Gender', 'DiscountPrice (in Rs)', 'OriginalPrice (in Rs)']]
cosine_sim = cosine_similarity(item_features, item_features)

# Function to get top recommendations based on category
def get_category_recommendations(category_id, num_recommendations=10):
    category_data = data_subset[data_subset['Category'] == category_id]
    if category_data.empty:
        return "No products found for the selected category."
    
    item_features = category_data[['BrandName', 'Category', 'Individual_category', 'category_by_Gender', 'DiscountPrice (in Rs)', 'OriginalPrice (in Rs)']]
    cosine_sim = cosine_similarity(item_features, item_features)
    
    # Assuming we use the first product as a reference for recommendation
    ref_index = category_data.index[0]
    content_scores = cosine_sim[ref_index]
    recommended_indices = np.argsort(content_scores)[-num_recommendations:]
    return category_data.iloc[recommended_indices]

# Streamlit UI
st.set_page_config(page_title="Trendvise - Myntra Product Recommendations", page_icon=":shirt:")
st.title('Trendvise')
st.subheader('Myntra Product Recommendation System')

# Map categories back to names for display
categories = data['Category'].unique()
category_names = {i: label_encoders['Category'].inverse_transform([i])[0] for i in categories}

st.sidebar.title('Category Selection')
selected_category = st.sidebar.selectbox('Choose a Category', options=[category_names[i] for i in categories])

if st.sidebar.button('Get Recommendations'):
    category_id = [k for k, v in category_names.items() if v == selected_category][0]
    recommendations = get_category_recommendations(category_id)
    
    if isinstance(recommendations, str):
        st.error(recommendations)
    else:
        st.success(f"Top {len(recommendations)} Product Recommendations for {selected_category}:")
        for index, row in recommendations.iterrows():
            st.markdown(f"### Product ID: {row['Product_id']}")
            st.write(f"**Brand:** {label_encoders['BrandName'].inverse_transform([row['BrandName']])[0]}")
            st.write(f"**Category:** {label_encoders['Category'].inverse_transform([row['Category']])[0]}")
            st.write(f"**Individual Category:** {label_encoders['Individual_category'].inverse_transform([row['Individual_category']])[0]}")
            st.write(f"**Category by Gender:** {label_encoders['category_by_Gender'].inverse_transform([row['category_by_Gender']])[0]}")
            st.write(f"**Discount Price:** ₹ {row['DiscountPrice (in Rs)']:.2f}")
            st.write(f"**Original Price:** ₹ {row['OriginalPrice (in Rs)']:.2f}")
            st.write(f"**Ratings:** {row['Ratings']:.1f}")
            st.write(f"**Reviews:** {row['Reviews']}")
            st.markdown("---")
