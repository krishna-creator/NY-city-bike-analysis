import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
try:
  basestring
except NameError:
  basestring = str
# run a sklearn model using (X_train, Y_train)
# and output the prediction on the X_test
def run(model,X_train, Y_train, X_test):
    model.fit(X_train, Y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)
    return y_pred

# use dummy variables to represent each categorical variable
def convert_categorical(X):
    columns = X.columns
    indices = X.index
    new_columns = []
    encoded_x = None
    for i in range(0, len(columns)):
        if X[columns[i]].dtype!='O': continue
        label_encoder = LabelEncoder()
        le = label_encoder.fit(X[columns[i]].apply(str))
        for class_ in le.classes_:
            new_columns.append("{}_{}".format(columns[i],class_))
        feature = le.transform(X[columns[i]].apply(str))
        feature = feature.reshape(X.shape[0], 1)
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoder.fit(feature)
        feature = onehot_encoder.transform(feature)
        if encoded_x is None:
            encoded_x = feature
        else:
            encoded_x = np.concatenate((encoded_x, feature), axis=1)
        X = X.drop(columns[i], axis=1)
    new_columns.extend(X.columns)
    X = pd.DataFrame(np.concatenate((encoded_x,X),axis=1),index=indices,columns=new_columns)
    return X

def preprocessing(dt):
    # convert pickup and dropoff time to datetime
    dt['lpep_dropoff_datetime'] = pd.to_datetime(dt['lpep_dropoff_datetime'],format= "%Y-%m-%d %H:%M:%S", errors='coerce')
    dt['lpep_pickup_datetime'] = pd.to_datetime(dt['lpep_pickup_datetime'], format = "%Y-%m-%d %H:%M:%S",errors='coerce')
    # get pickup/dropoff hour of the day
    dt['d_hour_of_day'] = dt['lpep_dropoff_datetime'].apply(lambda dt: dt.strftime('%H'))
    dt['p_hour_of_day'] = dt['lpep_pickup_datetime'].apply(lambda dt: dt.strftime('%H'))
    # get the duration of the trip based on seconds
    # we later use this to find the speed of taxi
    dt['pickup_dropoff_diff'] = dt['lpep_dropoff_datetime']-dt['lpep_pickup_datetime']
    dt['pickup_dropoff_diff'] = dt['pickup_dropoff_diff'].apply(lambda row: row.total_seconds())
    dt['pickup_dropoff_diff'] = np.log(dt['pickup_dropoff_diff'])
    # get dropoff day of the week
    dt['day_of_week'] = dt['lpep_dropoff_datetime'].apply(lambda row: row.weekday())
    # convert to categorical variables
    dt['store_and_fwd_flag'] = dt['store_and_fwd_flag'].astype(basestring)
    dt['d_hour_of_day'] = dt['d_hour_of_day'].astype(basestring)
    dt['p_hour_of_day'] = dt['p_hour_of_day'].astype(basestring)
    dt['day_of_week'] = dt['day_of_week'].astype(basestring)
    dt['RatecodeID'] = dt['RatecodeID'].astype(basestring)
    dt['payment_type'] = dt['payment_type'].astype(basestring)
    dt['trip_type'] = dt['trip_type'].astype(basestring)
    dt['VendorID'] = dt['VendorID'].astype(basestring)
    dt['PULocationID'] = dt['PULocationID'].astype(basestring)
    dt['DOLocationID'] = dt['DOLocationID'].astype(basestring)
    dt['passenger_count'] = dt['passenger_count'].astype(basestring)
    # add the dependent variable
    dt["tip_fair"] = dt['tip_amount']/dt['total_amount']
    dt['tip_fair_class']=np.nan
    dt['tip_fair_class'][dt['tip_fair']<0.2]=0
    dt['tip_fair_class'][dt['tip_fair']>=0.2]=1
    # remove junk column
    dt = dt.drop(['VendorID','tip_amount','ehail_fee','lpep_pickup_datetime','lpep_dropoff_datetime'],axis=1)
    return dt

def run_classifier(model,x_train,y_train,x_test):
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    return y_pred
