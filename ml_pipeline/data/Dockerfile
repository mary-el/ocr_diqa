FROM python:3.11

COPY requirements.txt data_load.py preprocess.py configs.py /workdir/
COPY /src /workdir/src
WORKDIR /workdir

RUN pip install -r requirements.txt
