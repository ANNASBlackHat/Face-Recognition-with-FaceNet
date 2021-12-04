# Face-Recognition-with-FaceNet
this app is using FaceNet (tflite version) model to recognize a face. You obviously be able to use the same model to do face recognition on mobile app.    

FaceNet: [A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)


 ![Demo App](https://github.com/ANNASBlackHat/Face-Recognition-with-FaceNet/raw/master/images/webcam%20test.png)


### Installation      
**Docker**     
```
docker build -t face-app .
docker run --rm -v ${PWD}:/app face-app
```

**Manual Installation**       
you can use virtual env if you prefer       
```
python3 -m venv venv
source venv/bin/active
```

then install all requirements      
```
pip install -r requirements.txt
```



### Run The App     
**Training**     
```
python webcam.py --label jokowi --max 5
```

after training, do face recognition:
```
python webcam.py
```
