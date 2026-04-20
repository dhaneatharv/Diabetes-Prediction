import pickle
from sklearn.metrics import r2_score

rf = pickle.load(open("models/rf.pkl", "rb"))
lr = pickle.load(open("models/lr.pkl", "rb"))
dt = pickle.load(open("models/dt.pkl", "rb"))

from preprocess import load_and_preprocess
X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

print("RF:", r2_score(y_test, rf.predict(X_test)))
print("LR:", r2_score(y_test, lr.predict(X_test)))
print("DT:", r2_score(y_test, dt.predict(X_test)))