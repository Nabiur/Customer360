import pickle
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu



# Sidebar for navigation
with st.sidebar:
	selected = option_menu(
		'Customer360 Analytics',
		['Home', 'Customer Churn Prediction', 'Customer Lifetime Value Prediction', 'Customer Segmentation'],
		icons=['house', 'person-x', 'currency-dollar', 'people-fill'],
		default_index=0
	)

# Home Page
if selected == 'Home':
    st.title('Retail Customer Analytics Machine Learning Application')
    st.image("retail.png", use_container_width=True,width=600)  # local image
    st.write("""
        Welcome to **Customer360**, a retail-focused machine learning application designed to help businesses understand and optimize customer behavior.

        This platform demonstrates how data science can be applied in the retail industry to improve customer engagement, maximize revenue, and support data-driven decision making.

        ### 🔑 Features of this App:
        - **Customer Churn Prediction** 🛑
          Identify customers who are at risk of leaving (no purchases in the last 6 months).
          This helps businesses proactively retain customers with targeted marketing campaigns.

        - **Customer Lifetime Value (CLV) Prediction** 💰
          Estimate the expected future value of each customer using purchase history.
          Enables smarter budgeting for customer acquisition and retention.

        - **Customer Segmentation (Unsupervised Learning)** 🎯
          Segment customers into meaningful groups (e.g., VIP loyalists, one-time buyers, bargain hunters).
          Helps design tailored promotions, product recommendations, and loyalty programs.

        - **Geographic Insights** 🌍
          Visualize customer distribution across countries to understand regional differences in behavior and value.

        - **Interactive Visualizations** 📊
          Explore patterns in purchasing behavior with charts, maps, and cluster profiling.

        ### 🎯 Business Impact:
        - Reduce churn by identifying at-risk customers early.
        - Increase profitability by focusing marketing on high-CLV customers.
        - Improve targeting through personalized offers for each customer segment.
        - Optimize inventory and promotions with insights from customer purchase patterns.

        ---
        This app is a **portfolio project** showcasing how real-world retail datasets can be transformed into actionable business insights using **machine learning** and **Streamlit**.
    """)

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
			# Define cluster personas + actions
			cluster_actions = {
				0: {
					"label": "Low-Value Regulars",
					"action": "Low-cost retention strategies (emails, basic offers)."
				},
				1: {
					"label": "VIP Loyalists",
					"action": "Loyalty programs, premium offers, early product access."
				},
				2: {
					"label": "One-Time Buyers / Churned",
					"action": "Likely not worth heavy investment → send only cheap retention (newsletter)."
				},
				3: {
					"label": "Infrequent VIP",
					"action": "Reactivation campaigns (exclusive deals, comeback discounts)."
				},
				4: {
					"label": "Active Mid-Tier",
					"action": "Upsell/cross-sell, personalized recommendations to push them to VIP."
				}
			}

			# Get the predicted cluster from model
			pred_cluster = int(input_df['predicted_cluster'].iloc[0])

			# Map to label + action
			segmentation_label = cluster_actions[pred_cluster]["label"]
			segmentation_action = cluster_actions[pred_cluster]["action"]

			# Show result in Streamlit
			st.subheader("🎯 Customer Segment Result")
			st.write(f"**Cluster {pred_cluster} → {segmentation_label}**")
			st.success(f"Recommended Action: {segmentation_action}")



		except Exception as e:
			segmentation_result = f"Error: {e}"

	st.success(segmentation_result)





