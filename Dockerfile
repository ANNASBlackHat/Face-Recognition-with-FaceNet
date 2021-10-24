FROM python:3.9-slim

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app
COPY . .

RUN install virtualenv
RUN virtualenv /app/venv
RUN source /app/venv/bin/activate
RUN pip install -r requirements.txt


ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN . venv/bin/activate
CMD [ "/app/venv/bin/python", "-u", "webcam.py" ]
