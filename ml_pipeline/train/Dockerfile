FROM python:3.11

COPY /train/ /workdir/train
COPY /utils/ /workdir/utils
COPY configs.py /workdir
WORKDIR /workdir

RUN pip install -r train/requirements.txt
