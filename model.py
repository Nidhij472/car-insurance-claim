import numpy as np
import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

df=pd.read_csv('C:/Users/nidhi-jaiswal/Downloads/train.csv/train.csv')
df_featured=df.copy()
df_featured['feature_score']=df[['is_esc', 'is_adjustable_steering', 'is_tpms', 'is_parking_sensors',
       'is_parking_camera','is_front_fog_lights',
       'is_rear_window_wiper', 'is_rear_window_washer',
       'is_rear_window_defogger', 'is_brake_assist', 'is_power_door_locks',
       'is_central_locking', 'is_power_steering',
       'is_driver_seat_height_adjustable', 'is_day_night_rear_view_mirror',
       'is_ecw', 'is_speed_alert']].replace(['Yes','No'],[1,0]).astype('int64').sum(axis=1)
df_featured.drop(['is_esc', 'is_adjustable_steering', 'is_tpms', 'is_parking_sensors',
       'is_parking_camera','is_front_fog_lights',
       'is_rear_window_wiper', 'is_rear_window_washer',
       'is_rear_window_defogger', 'is_brake_assist', 'is_power_door_locks',
       'is_central_locking', 'is_power_steering',
       'is_driver_seat_height_adjustable', 'is_day_night_rear_view_mirror',
       'is_ecw', 'is_speed_alert'],axis=1,inplace=True)
df_featured['volume']=df_featured['length'] * df_featured['width'] * df_featured['height']
df_featured.drop(['length','width','height'],axis=1,inplace=True)
def final_max_torque(x):
    p=x.split('bhp@')[0]
    r = x.split('bhp@')[1].strip('rpm')
    p_float = float(p)
    r_float = float(r)
    return (p_float*5252*1.3558179483/6000)
def final_max_power(x):
    t=x.split('Nm@')[0]
    r = x.split('Nm@')[1].strip('rpm')
    t_float = float(t)
    r_float = float(r)
    return ((t_float*r_float)/(1.3558179483*5252))
df_featured['final_max_power']=df_featured.max_torque.apply(final_max_power)
df_featured['final_max_torque']=df_featured.max_power.apply(final_max_torque)
df_featured.drop(['max_torque','max_power'],axis=1,inplace=True)
df_featured.drop('policy_id',axis=1,inplace=True)
X=df_featured.drop('is_claim',axis=1)
y=df_featured.is_claim.astype(int)
X=pd.get_dummies(X,drop_first=True)
smn = SMOTE()
X2, y2 = smn.fit_resample(X, y)
xtrain4, xtest4, ytrain4, ytest4 = train_test_split(X2, y2, test_size=0.3, random_state=48)
xgb=XGBClassifier()
final_model=xgb.fit(xtrain4,ytrain4)
ypred=final_model.predict(xtrain4)
ypred1=final_model.predict(xtest4)
print(classification_report(ytrain4,ypred))
print(classification_report(ytest4,ypred1))
