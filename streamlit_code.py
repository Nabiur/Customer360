import pickle
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu



# Sidebar for navigation
with st.sidebar:
	selected = option_menu(
		'Retail Customer Analytics',
		['Customer Churn Prediction', 'Customer Lifetime Value Prediction', 'Customer Segmentation'],
		icons=['person-x', 'currency-dollar', 'people-fill'],
		default_index=0
	)

# Customer Churn Prediction Page
if selected == 'Customer Churn Prediction':
	with open("customer_churn_model.pkl", "rb") as f:
		model_data=pickle.load(f)

	model=model_data['model']
	features=model_data['features_names']
	st.title('Customer Churn Prediction using ML')

	# Input fields
	col1, col2, col3 = st.columns(3)

	with col1:
		recency = int(st.text_input('Recency (days since last purchase)') or 0)
	with col2:
		frequency = int(st.text_input('Frequency (number of purchases)') or 0)
	with col3:
		monetary = float(st.text_input('Monetary (total spending)') or 0.0)
	with col1:
		avg_basket = float(st.text_input('Average Basket Value') or 0.0)
	with col2:
		diversity = int(st.text_input('Diversity (number of categories purchased)') or 0)
	with col3:
		tenure = int(st.text_input('Tenure (days since first purchase)') or 0)
	with col1:
			country = st.selectbox('Country', [
			'United Kingdom', 'France', 'Australia', 'Netherlands', 'Germany',
			'Norway', 'EIRE', 'Switzerland', 'Spain', 'Poland', 'Portugal',
			'Italy', 'Belgium', 'Lithuania', 'Japan', 'Iceland',
			'Channel Islands', 'Denmark', 'Cyprus', 'Sweden', 'Finland',
			'Austria', 'Greece', 'Singapore', 'Lebanon',
			'United Arab Emirates', 'Israel', 'Saudi Arabia', 'Czech Republic',
			'Canada', 'Unspecified', 'Brazil', 'USA', 'European Community',
			'Bahrain', 'Malta', 'RSA'
		])

	# Prediction
	churn_prediction = ''
	if st.button('Predict Churn'):
		try:
			input_data = {
				'Recency':recency,
				'frequency':frequency,
				'Monetary':monetary,
				'AvgBasket':avg_basket,
				'Diversity':diversity,
				'Tenure':tenure,
				'Country':country
			}
			input_df = pd.DataFrame([input_data])

			with open('churn_encoders.pkl','rb') as f:
				encoders=pickle.load(f)

			for col, encoder in encoders.items():
				if col in input_df.columns:
					input_df[col] = encoder.transform(input_df[col])

			prediction = model.predict(input_df)
			churn_prediction = 'Customer will Churn' if prediction[0] == 1 else 'Customer will Not Churn'

		except Exception as e:
			churn_prediction = f"Error: {e}"

	st.success(churn_prediction)

# Customer Lifetime Value Prediction Page
if selected == 'Customer Lifetime Value Prediction':
	with open("CLV_prediction_model.pkl", "rb") as f:
		model_data=pickle.load(f)

	model=model_data['model']
	features=model_data['features_names']
	st.title('Customer Lifetime Value Prediction using ML')

	# Input fields
	col1, col2,col3 = st.columns(3)

	with col1:
		recency = int(st.text_input('Recency (days since last purchase)') or 0)
	with col2:
		frequency = int(st.text_input('Frequency (number of purchases)') or 0)
	with col3:
		monetary = float(st.text_input('Monetary (total spending)') or 0.0)
	with col1:
		avg_basket = float(st.text_input('Average Basket Value') or 0.0)
	with col2:
		diversity = int(st.text_input('Diversity (number of categories purchased)') or 0)
	with col3:
		tenure = int(st.text_input('Tenure (days since first purchase)') or 0)
	with col1:
			country = st.selectbox('Country', [
			'United Kingdom', 'France', 'Australia', 'Netherlands', 'Germany',
			'Norway', 'EIRE', 'Switzerland', 'Spain', 'Poland', 'Portugal',
			'Italy', 'Belgium', 'Lithuania', 'Japan', 'Iceland',
			'Channel Islands', 'Denmark', 'Cyprus', 'Sweden', 'Finland',
			'Austria', 'Greece', 'Singapore', 'Lebanon',
			'United Arab Emirates', 'Israel', 'Saudi Arabia', 'Czech Republic',
			'Canada', 'Unspecified', 'Brazil', 'USA', 'European Community',
			'Bahrain', 'Malta', 'RSA'
		])

	# Prediction
	clv_prediction = ''
	if st.button('Predict CLV'):
		try:
			input_data = {
				'Recency':recency,
				'frequency':frequency,
				'Monetary':monetary,
				'AvgBasket':avg_basket,
				'Diversity':diversity,
				'Tenure':tenure,
				'Country':country
			}
			input_df = pd.DataFrame([input_data])

			with open('clv_encoders.pkl','rb') as f:
				encoders=pickle.load(f)

			for col, encoder in encoders.items():
				if col in input_df.columns:
					input_df[col] = encoder.transform(input_df[col])
			prediction = model.predict(input_df)
			clv_prediction = f'Predicted Customer Lifetime Value: ${prediction[0]:.2f}'

		except Exception as e:
			clv_prediction = f"Error: {e}"

	st.success(clv_prediction)

# Customer Segmentation Page

if selected == 'Customer Segmentation':
	with open("Customer_segmentation_model.pkl", "rb") as f:
		model_data=pickle.load(f)

	model=model_data['model']
	features=model_data['features_names']
	st.title('Customer Segmentation using ML')

	# Input fields
	col1, col2, col3 = st.columns(3)
	with col1:
		recency = int(st.text_input('Recency (days since last purchase)') or 0)
	with col2:
		frequency = int(st.text_input('Frequency (number of purchases)') or 0)
	with col3:
		monetary = float(st.text_input('Monetary (total spending)') or 0.0)
	with col1:
		avg_basket = float(st.text_input('Average Basket Value') or 0.0)
	with col2:
		diversity = int(st.text_input('Diversity (number of categories purchased)') or 0)
	with col3:
		tenure = int(st.text_input('Tenure (days since first purchase)') or 0)
	with col1:
		country = st.selectbox('Country', [
			'United Kingdom', 'France', 'Australia', 'Netherlands', 'Germany',
			'Norway', 'EIRE', 'Switzerland', 'Spain', 'Poland', 'Portugal',
			'Italy', 'Belgium', 'Lithuania', 'Japan', 'Iceland',
			'Channel Islands', 'Denmark', 'Cyprus', 'Sweden', 'Finland',
			'Austria', 'Greece', 'Singapore', 'Lebanon',
			'United Arab Emirates', 'Israel', 'Saudi Arabia', 'Czech Republic',
			'Canada', 'Unspecified', 'Brazil', 'USA', 'European Community',
			'Bahrain', 'Malta', 'RSA'
		])
	# Prediction
	segmentation_result = ''
	if st.button('Segment Customer'):
		try:
			input_data = {
				'Recency':recency,
				'frequency':frequency,
				'Monetary':monetary,
				'AvgBasket':avg_basket,
				'Diversity':diversity,
				'Tenure':tenure,
				'Country':country
			}
			input_df = pd.DataFrame([input_data])
			input_df.drop('Country', axis=1, inplace=True)
			with open('segment_scalers.pkl','rb') as f:
				scalers=pickle.load(f)
			scaler = scalers['scale']
			scaled_df = pd.DataFrame(scaler.transform(np.log1p(input_df)), columns=input_df.columns)


			predicted_clusters = model.predict(scaled_df)
			input_df['predicted_cluster'] = predicted_clusters
			segmentation_result = input_df['predicted_cluster'].replace({0: "Low-Value Regulars", 1: "VIP Loyalists", 2: "Churned", 3: "Infrequent VIP", 4: "Active Mid-Tier"}).iloc[0]


		except Exception as e:
			segmentation_result = f"Error: {e}"

	st.success(segmentation_result)





