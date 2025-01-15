import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_model():
    with open('lgbm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model= load_model()

path= 'Training_data'
df= pd.read_csv('vestiaire.csv')
X_train= df.drop(['price_usd'], axis=1)

brands= list(X_train['brand_name'].unique())
seller_country= list(X_train['seller_country'].unique())
types= list(X_train['product_type'].unique())

st.markdown("<h1 style='text-align: center;'>Price Prediction Centre</h1>", unsafe_allow_html=True)

st.write('Enter the required details to get the predicted price')

product_type = st.selectbox('Product Type',types)
brand = st.selectbox('Brand Name',brands)
seller = st.selectbox('From where does the seller belong to?',seller_country)
buyers_fees= st.number_input('Fees paid extra by buyer',min_value=0.0, step=0.01)
seller_followers= st.number_input('Number of followers of seller on the platform',min_value=0, step=1)
seller_products_listed = st.number_input('Number of products available from the seller on the platform',min_value=0, step=1)
seller_products = st.number_input('Number of products Sold by the seller on the platform',min_value=0, step=1)
    
if st.button('Submit'):
    user_input = pd.DataFrame({
        'product_type': [product_type],
        'brand_name': [brand],
        'seller_country': [seller],
        'buyers_fees': [buyers_fees],
        'seller_num_followers': [seller_followers],
        'seller_listed_products_for_sale':[seller_products_listed],
        'seller_products_sold':[seller_products]
    })

    def log_transform_with_shift(df, columns):

        for column in columns:
            if pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column]):  
                if (df[column] <= 0).any():
                    shift_value = abs(df[column].min()) + 1
                    df[column] = df[column] + shift_value

                df[column] = np.log(df[column])
    
        return df
    
    user_input= log_transform_with_shift(user_input,user_input.columns)

    # Label encoding: Apply separately for each categorical column
    categorical_cols = ['brand_name', 'product_type', 'seller_country']
    label_encoders = {}

    # Fit the encoders during training phase
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(X_train[col])  # Fit the encoder to the training data
        label_encoders[col] = le

    # Encode user input for each categorical column
    for col in categorical_cols:
        user_input[col] = label_encoders[col].transform(user_input[col])

    numerical_cols = ['buyers_fees', 'seller_num_followers', 'seller_listed_products_for_sale', 'seller_products_sold']

    scaler = StandardScaler()
    scaler.fit(X_train[numerical_cols])

    user_input[numerical_cols] = scaler.transform(user_input[numerical_cols])

    # Ensure that all columns in user_input match X_train, adding missing columns as 0
    missing_cols = set(X_train.columns) - set(user_input.columns)
    for col in missing_cols:
        user_input[col] = 0

    # Reorder the user input columns to match X_train's order
    user_input = user_input[X_train.columns]

    user_input= user_input.drop(['Unnamed: 0'],axis=1)

    log_prediction= model.predict(user_input)
    prediction = np.exp(log_prediction)
    st.write('The predicted price in usd is: ',prediction)
