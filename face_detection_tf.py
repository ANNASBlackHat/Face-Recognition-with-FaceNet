from mtcnn import MTCNN
import tensorflow as tf
import cv2
from tinydb import TinyDB, Query
from tinydb.operations import add, set
import time
import os
import numpy as np


detector = MTCNN()
model = tf.lite.Interpreter('model/mobile_facenet.tflite')
model.allocate_tensors()

input_detail = model.get_input_details()
input_shape = input_detail[0]['shape']
input_index = input_detail[0]['index']
output_index = model.get_output_details()[0]['index']
shape = (input_shape[1], input_shape[2])

print('input detail: ', input_detail)


os.makedirs(f'temp/db', exist_ok=True)
db = TinyDB('temp/db/database.json')
users = db.table('users').all()
all_embeddings = []
labels = []
for user in users:
    all_embeddings.extend(user['embeddings'])
    labels.extend([user['label'] for x in range(len(user['embeddings']))])


def millis():
    return round(time.time() * 1000)


def detect_face(img):
    detector.detect_faces(img)
    # print(f'{len(faces)} detected')
    # for face in faces:
    #     print(f'box {face["box"]}')


def process_image(img, required_size=(160, 160)):
    ret = cv2.resize(img, required_size)
    ret = ret.astype('float32')
    mean, std = ret.mean(), ret.std()
    ret = (ret - mean) / std
    return ret


def find_nearest(face_emb):
    distance = 4.0
    nearest_index = -1
    for i, emb in enumerate(all_embeddings):
        cal = np.linalg.norm(face_emb - emb)
        if cal < distance:
            distance = cal
            nearest_index = i
    return nearest_index, distance


def save_trainig(embs, label):
    emb = [embs[0].tolist()]
    q = Query()
    users = db.table('users').search(q.label == label)
    if len(users) == 0:
        db.table('users').insert(
            {'label': label,  'id': str(millis()), 'embeddings': emb})
        print(f'new label added: {label} ')
    else:
        e = users[0]['embeddings']
        e.extend(emb)
        db.table('users').update(set('embeddings', e), q.label == label)
        print(f'{label} added to existing data')


def recognize(img, training=False, label=''):
    input = process_image(img, shape)
    input = input.reshape(input_shape)
    print('input index', input_index)
    print('shape: ', shape)
    print('input shape: ', input.shape)
    try:
        model.set_tensor(input_index, input)

        model.invoke()
        emb = model.get_tensor(output_index)
        # print('EMB -->', emb)

        if training:
            print('----- saving embedding -------')
            save_trainig(emb, label)
            return label, 100
        else:
            index, distance = find_nearest(emb)
            print(f'------> found : {index} ({labels[index]}) | distance: {distance}')
            if distance > 1.1 :
                return '[unknown]', 0
            return labels[index], 100.0 - (distance * 100.0)
    except Exception as e:
        print('errr', e)
