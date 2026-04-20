import os
os.makedirs("models", exist_ok=True)
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from preprocess import load_and_preprocess
X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

rf = RandomForestRegressor(n_estimators=100)
lr = LinearRegression()
dt = DecisionTreeRegressor(max_depth=5)

rf.fit(X_train, y_train)
lr.fit(X_train, y_train)
dt.fit(X_train, y_train)

# save models
pickle.dump(rf, open("models/rf.pkl", "wb"))
pickle.dump(lr, open("models/lr.pkl", "wb"))
pickle.dump(dt, open("models/dt.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))

print("Models trained & saved")
