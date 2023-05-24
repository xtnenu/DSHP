import numpy as np
import shap
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import load_model
from utils import loadcsv
from sklearn.model_selection import StratifiedKFold
from tensorflow import compat
from tensorflow.keras.optimizers import SGD

#model=load_model("v2_new0.h5")
#SGDN=SGD(learning_rate=0.0005)
#model.compile(loss='binary_crossentropy', optimizer=SGDN, metrics=['accuracy'])
X=loadcsv("Data.csv",",")
X = np.array(X)
X = X.astype("float32")
Y = X[:, -1]
X = X[:, :-1]
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
for train, test in  kfold.split(X, Y):
    n=0
    model=load_model("v2_new0.h5")
    SGDN=SGD(learning_rate=0.0005)
    model.compile(loss='binary_crossentropy', optimizer=SGDN, metrics=['accuracy'])
    new=X[train].reshape(-1,len(X[train][0]),1)
    explainer = shap.Explainer(model.predict, new)
    shap_values = explainer(new[0])
    # Plot the SHAP values
    shap.plots.waterfall(shap_values)
    #explainer=shap.DeepExplainer(model,new[:100])
    #sample = new[0:1, :]
    #shap_values, expected_value = explainer.shap_values(sample)
    #shap_values= explainer.shap_values(sample)
    #print(shap_values.shape)

# Visualize the explanations
    #shap.summary_plot(shap_values, features=sample)
"""
# Load the Boston Housing dataset
X,y = shap.datasets.boston()

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Compute SHAP values for a single instance
explainer = shap.Explainer(model.predict, X)
shap_values = explainer(X[0])

# Plot the SHAP values
shap.plots.waterfall(shap_values)
"""