import os   # file access
import pandas as pd # csv/IO processing
import category_encoders as ce
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB

current_working_directory = os.getcwd()
file = r'\input\Thyroid_Diff.csv'
data = current_working_directory + file

df = pd.read_csv(data, header=0,sep=',')

targetVariable = 'Recurred'
X = df.drop([targetVariable], axis=1)

y = df[targetVariable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

encoder = ce.OneHotEncoder(cols=categorical)
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

cols = X_train.columns

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

model = GaussianNB()
model.fit(X_train, y_train)

joblib.dump(model, "gb_model.sav")