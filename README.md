# Face-Recognition-with-FaceNet
this app is using FaceNet (tflite version) model to recognize a face. You obviously be able to use the same model to do face recognition on mobile app.    

FaceNet: [A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)


 ![Demo App](https://github.com/ANNASBlackHat/Face-Recognition-with-FaceNet/raw/master/images/webcam%20test.png)


### Run The App     
**Training**
```
python webcam.py --label jokowi --max 5
```

after training, do face recognition:
```
python webcam.py
```
