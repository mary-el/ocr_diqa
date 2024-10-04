FROM python:3.11

COPY /predict/ /workdir/predict
COPY /utils/ /workdir/utils
COPY configs.py /workdir
WORKDIR /workdir

RUN pip install -r predict/requirements.txt
