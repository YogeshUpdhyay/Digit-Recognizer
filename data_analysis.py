import numpy as np
import pandas as pd 
import matplotlib.pyplot as pyplot
from PIL import Image


train_data_path =  'digit-recognizer/train.csv'
test_data_path  =   'digit-recognizer/test.csv'
validation_data_path = 'digit-recognizer/sample_submission.csv'
dt = pd.read_csv(test_data_path)
df = pd.read_csv(train_data_path)
#print(df.head())
dv = pd.read_csv(validation_data_path)

labels = df.iloc[:,0]
pixel_values = df.iloc[:,1:]
X_test = dt.iloc[:,0:]
y_test = dv.iloc[:,1]

image_arrays,test_image_array = list(),list()


for i in range(len(pixel_values)):
    pixels = np.array(pixel_values.iloc[i])
    pixels = np.reshape(pixels,(28,28))
    image_arrays.append(pixels)

for i in range(len(X_test)):
    pixels = np.array(X_test.iloc[i])
    pixels = np.reshape(pixels,(28,28))
    test_image_array.append(pixels)

"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(image_arrays)
X_test = sc.transform(test_image_array)
"""



from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(image_arrays, labels)

y_pred = classifier.predict(test_image_array)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("done")