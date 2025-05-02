#Below are just how the data is layed out, will probably delete later.
#id: Unique identifer for transaction
#V1-V28: Anonomyized features, various attributes (time, location etc.)
#Amount: the transaction amount
#Class: Binary Label indicating whether transaction is fraudulent or not. Binary 1 for and 0 for not


#The notebook - fraud_detection.ipynb is where I created both of the models
# There is also just some stuff in there where I was playing around
#trying to see things about the dataset. Models folder holds the models


import numpy as np
import joblib 
import pandas as pd


#This function takes in data (V1-28 and amount ONLY), a model, and a scaler. It returns the 
#   prediction of a singular transaction. Multiple is possible, just need to change some lines
#   More specifically we'd have to change the way we frame df_input. Right now its only 1x29,
#   but it could be nx29 in theory
def predict_fraud(data, model, scaler):

    #Frame data into a DataFrame. This avoids warning i was getting earlier -V
    feature_names = [f"V{i}" for i in range(1,29)] + ["Amount"]
    model_input = pd.DataFrame(data, columns=feature_names)
    model_input['Amount'] = scaler.transform(model_input[['Amount']])
    model_input.rename(columns={'Amount': 'scaled_amount'}, inplace =True)

    

    #####OLD CODE had warnings but works. This is like barebones functionality######
    # data = np.array(data)
    # amount = data[:, -1].reshape(-1, 1)  # last column is unscaled amount
    # scaled_amount = scaler.transform(amount)
    # # Replace the last column with the scaled version
    # data[:, -1] = scaled_amount.flatten()

    # Predict
    return model.predict(model_input)



def main():
    #model1 takes approximatley 5 mins to train.
    #So we can just load it from file
    model1 = joblib.load('models/random_forest_fraud_model.pkl')
    
    #model2 takes around 1 second to train. 
    model2 = joblib.load('models/logistic_regression_fraud_model.pkl')
    scaler = joblib.load('models/scaler.pkl')

    #transaction with ID 2
    legit_transaction = [[
    -0.260271613,	-0.949384607,	1.7285377761514877,
    -0.457986289,	0.074061654,	1.4194811432767418,
    0.7435110747693963,	-0.095576013,	-0.261296619,	
    0.6907077998548777,	-0.272984925,	0.6592006642046833,
    0.8051731885973652,	0.6168743863580851,	3.0690247739919467,
    -0.577513522,	0.8865259684145712,	0.239441661,
    -2.366078928,	0.3616523101955929,	-0.005020278,
    0.7029063846645285,	0.9450454906841043,	-1.154665629,
    -0.605563661,	-0.312894548,	-0.300258035,	-0.244718229,	#V1-V28
    2513.54	 #Raw amount
    ]]

    #transaction with ID 529557
    fraud_transaction = [[
    	-2.022894901,	2.630825326275901,	-1.929035196,
        2.2600152592563405,	-2.356845976,	0.2984466949522401,
        -2.812206568,	-1.154011776,	-2.540819974,
        -2.340405008,	1.2680027163160732,	-1.697174611,
        -0.170531078,	-0.998484346,	-1.139099244,
    	-1.634768711,	-1.753095717,	-1.734032017,
        1.0954770255575201,	2.181779210129651,	-2.143351577,
        2.989039784843061,	1.6549870353987193,	0.8008703761032576,
        -0.520501302,	0.8648731646214095,	-2.843501834,	-1.421128844,
        13595.26
    ]]
    


    result = predict_fraud(fraud_transaction,model1, scaler)
    print("Model 1 Prediction with Fraud transaction: ", "FRAUD" if result[0] == 1 else "LEGIT")

    result2 = predict_fraud(legit_transaction,model1,scaler)
    print("Model 1 Prediction with Legit transaction: ", "FRAUD" if result2[0] == 1 else "LEGIT") 

    result3 = predict_fraud(fraud_transaction,model2, scaler)
    print("Model 2 Prediction with Fraud transaction: ", "FRAUD" if result3[0] == 1 else "LEGIT")

    result4 = predict_fraud(legit_transaction,model2,scaler)
    print("Model 2 Prediction with Legit transaction: ", "FRAUD" if result4[0] == 1 else "LEGIT") 



main()