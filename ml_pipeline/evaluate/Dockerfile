FROM python:3.11

COPY /evaluate /workdir/evaluate/
COPY /utils /workdir/utils/
COPY configs.py /workdir
WORKDIR /workdir

RUN pip install -r evaluate/requirements.txt
