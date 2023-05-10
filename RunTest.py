import pandas as pd
import cv2
import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf



model=tf.keras.models.load_model('Model600.tfl')
def predict():
    predictions = []
    for image in tqdm(os.listdir('TestSamples')):
        path = os.path.join('TestSamples', image)
        img_data = cv2.imread(path)
        img_data = cv2.resize(img_data, (224, 224)).reshape(-1,224,224,3)

        idx = 0
        value = 0
        cnt = 0
        p = model.predict([np.array(img_data)])[0]
        for j in p:
            if j > value:
                value = j
                idx = cnt
            cnt += 1
        predictions.append([image, idx])
    return predictions

predictions = predict()
labels = pd.DataFrame(predictions, columns=['image_name', 'label'])
labels.to_csv('result1.csv', index=False)

#results = model.evaluate(x_test, y_test, batch_size=128)
#print("test loss, test acc:", results)