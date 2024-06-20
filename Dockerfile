FROM python:3.6

WORKDIR /opt/detect

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/install_onnxruntime.sh ./
RUN ./install_onnxruntime.sh

COPY detect detect
COPY models models

CMD [ "python3", "-m", "detect" ]
