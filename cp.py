import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('agridata333.csv')


X = data.drop('label', axis=1)
y = data['label']


le = LabelEncoder()
y = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gnb = GaussianNB()


gnb.fit(X_train, y_train)

pickle.dump(gnb,open('cp.pkl','wb'))
cp=pickle.load(open('cp.pkl','rb'))

def get_crop_recommendation():
    N = int(input("Enter the Nitrogen content (in kg/ha): "))
    P = int(input("Enter the Phosphorus content (in kg/ha): "))
    K = int(input("Enter the Potassium content (in kg/ha): "))
    temperature = float(input("Enter the average temperature (in °C): "))
    humidity = float(input("Enter the relative humidity (in %): "))
    ph = float(input("Enter the soil pH value: "))
    rainfall = float(input("Enter the rainfall (in mm): "))

   
    user_input = [[N, P, K, temperature, humidity, ph, rainfall]]

    predicted_crop = gnb.predict(user_input)

    
    predicted_crop = le.inverse_transform(predicted_crop)

    print(f"\nBased on the given inputs, the recommended crop is: {predicted_crop[0]}")


get_crop_recommendation()
