
import json, os, re, sys, time
import numpy as np

from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image


def predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds


CLASS_INDEX = None
JSON_PATH = "idenprof_model_class.json"


def decode_predictions(preds, top=5, model_json=""):
    global CLASS_INDEX

    if CLASS_INDEX is None:
       CLASS_INDEX = json.load(open(model_json))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        for i in top_indices:
              each_result = []
              each_result.append(CLASS_INDEX[str(i)])
              each_result.append(pred[i])
              results.append(each_result)

    return results



if __name__ == '__main__':
    model_path = "resnet50_best.h5"
    print('Loading model:', model_path)
    t0 = time.time()
    model = load_model(model_path)
    t1 = time.time()
    print('Loaded in:', t1-t0)

    test_path = "7032-2.jpg"
    print('Generating predictions on image:', test_path)
    preds = predict(test_path, model)
    predictiondata = decode_predictions(preds, top=int(1), model_json=JSON_PATH)
    print(predictiondata)
    #model.summary()