import json
import pickle
import pandas as pd
import numpy as np
import streamlit as st

# Load model
with open('list_num_cols_n.txt', 'r') as file_1:
    loaded_num_cols_n = json.load(file_1)

with open('list_cat_cols_n.txt', 'r') as file_2:
    loaded_cat_cols_n = json.load(file_2)

with open('list_cat_cols_o.txt', 'r') as file_3:
    loaded_cat_cols_o = json.load(file_3)

with open('scaler_minmax.pkl', 'rb') as file_4:
    loaded_scaler_minmax = pickle.load(file_4)

with open('encoder_ordinal.pkl', 'rb') as file_5:
    loaded_encoder_ordinal = pickle.load(file_5)

with open('encoder_onehot.pkl', 'rb') as file_6:
    loaded_encoder_onehot = pickle.load(file_6)

with open('model_logreg_tuned.pkl', 'rb') as file_7:
    loaded_best_logreg_model = pickle.load(file_7)


# Form Input
def run():
    st.title('Credit Card Payment Default Prediction')

    with st.form(key='form_credit_card'):
        st.write('### Customer Information:')
        limit_balance = st.number_input('Limit Balance', min_value=0, max_value=1000000, value=50000)
        sex = st.selectbox('Sex (1 = Male, 2 = Female)', [1, 2])
        education_level = st.selectbox('Education Level (1 = Graduate School, 2 = University, 3 = High School, 4 = Others)', [1, 2, 3, 4])
        marital_status = st.selectbox('Marital Status (1 = Married, 2 = Single, 3 = Others)', [1, 2, 3])
        age = st.number_input('Age', min_value=20, max_value=80, value=30)
        
        st.markdown('---')
        
        st.write('### Payment Status:')
        st.caption('''**Indicates how early or late the customer made their repayment.**''')
        st.caption("""
        **Description:**  
        - -1  = pay duly (paid on time)  
        - 0   = payment in time  
        - 1   = payment delayed for one month  
        - 2   = payment delayed for two months and so on...
        """)

        pay_1 = st.selectbox('Pay Status in September', ('-2', '-1', '0', '1', '2', '3', '4', '5', '6'), index=1)
        pay_2 = st.selectbox('Pay Status in August', ('-2', '-1', '0', '1', '2', '3', '4', '5', '6'), index=1)
        pay_3 = st.selectbox('Pay Status in July', ('-2', '-1', '0', '1', '2', '3', '4', '5', '6'), index=1)
        pay_4 = st.selectbox('Pay Status in June', ('-2', '-1', '0', '1', '2', '3', '4', '5', '6'), index=1)
        pay_5 = st.selectbox('Pay Status in May', ('-2', '-1', '0', '1', '2', '3', '4', '5', '6'), index=1)
        pay_6 = st.selectbox('Pay Status in April', ('-2', '-1', '0', '1', '2', '3', '4', '5', '6'), index=1)
        
        st.markdown('---')

        st.write('### Bill Statement Ammount:')
        st.caption('''**The total amount billed to the customer for that period.**''')
        bill_amt_1 = st.number_input('Bill Statement Amount in September', min_value=0, max_value=1000000, value=10000)
        bill_amt_2 = st.number_input('Bill Statement Amount in August', min_value=0, max_value=1000000, value=10000)
        bill_amt_3 = st.number_input('Bill Statement Amount in July', min_value=0, max_value=1000000, value=10000)
        bill_amt_4 = st.number_input('Bill Statement Amount in June', min_value=0, max_value=1000000, value=10000)
        bill_amt_5 = st.number_input('Bill Statement Amount in May', min_value=0, max_value=1000000, value=10000)
        bill_amt_6 = st.number_input('Bill Statement Amount in April', min_value=0, max_value=1000000, value=10000)
        
        st.markdown('---')
        
        st.write('### Payment Ammount:')
        st.caption('''**The actual amount paid by the customer during that period.**''')
        pay_amt_1 = st.number_input('Payment Amount in September', min_value=0, max_value=1000000, value=5000)
        pay_amt_2 = st.number_input('Payment Amount in August', min_value=0, max_value=1000000, value=5000)
        pay_amt_3 = st.number_input('Payment Amount in July', min_value=0, max_value=1000000, value=5000)
        pay_amt_4 = st.number_input('Payment Amount in June', min_value=0, max_value=1000000, value=5000)
        pay_amt_5 = st.number_input('Payment Amount in May', min_value=0, max_value=1000000, value=5000)
        pay_amt_6 = st.number_input('Payment Amount in April', min_value=0, max_value=1000000, value=5000)

        st.markdown('---')    
        submitted = st.form_submit_button('Predict')

    data_inf = {
        'limit_balance': limit_balance,
        'sex': sex,
        'education_level': education_level,
        'marital_status': marital_status,
        'age': age,
        'pay_1': int(pay_1),
        'pay_2': int(pay_2),
        'pay_3': int(pay_3),
        'pay_4': int(pay_4),
        'pay_5': int(pay_5),
        'pay_6': int(pay_6),
        'bill_amt_1': bill_amt_1,
        'bill_amt_2': bill_amt_2,
        'bill_amt_3': bill_amt_3,
        'bill_amt_4': bill_amt_4,
        'bill_amt_5': bill_amt_5,
        'bill_amt_6': bill_amt_6,
        'pay_amt_1': pay_amt_1,
        'pay_amt_2': pay_amt_2,
        'pay_amt_3': pay_amt_3,
        'pay_amt_4': pay_amt_4,
        'pay_amt_5': pay_amt_5,
        'pay_amt_6': pay_amt_6
    }

    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submitted:
        # Numeric-Categoric split
        data_inf_num = data_inf[loaded_num_cols_n]
        data_inf_cat_o = data_inf[loaded_cat_cols_o]
        data_inf_cat_n = data_inf[loaded_cat_cols_n]

        # Scaling
        data_inf_num_scaled = loaded_scaler_minmax.transform(data_inf_num[['limit_balance']])

        # Encoding
        data_inf_cat_o_encoded = loaded_encoder_ordinal.transform(data_inf_cat_o)
        data_inf_cat_n_encoded = loaded_encoder_onehot.transform(data_inf_cat_n).toarray()

        # Combine all features
        data_inf_final = np.concatenate([data_inf_num_scaled, data_inf_cat_o_encoded, data_inf_cat_n_encoded], axis=1)
        

        # Prediction
        st.write('### Data for Prediction')
        prediction = loaded_best_logreg_model.predict(data_inf_final)
        
        st.write("Prediction Result:")
        if prediction[0] == 1:
            st.error("⚠️ The customer is predicted to be **at risk of credit card payment default**.")
        else:
            st.success("✅ The customer is predicted to have a **low risk of defaulting** on their payment.")


if __name__ == '__main__':
    run()