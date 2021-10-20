# Face-Recognition-with-FaceNet
this app is using FaceNet (tflite version) model to recognize a face. You obviously be able to use the same model to do face recognition on mobile app.    

FaceNet: [A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)

### Run The App     
**Training**
```
python webcam.py --label jokowi --max 5
```

after training, do face recognition:
```
python webcam.py
```
