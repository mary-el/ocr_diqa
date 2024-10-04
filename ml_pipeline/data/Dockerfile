FROM python:3.11

COPY data /workdir/data
COPY utils /workdir/utils
COPY configs.py /workdir
WORKDIR /workdir

RUN pip install -r data/requirements.txt
