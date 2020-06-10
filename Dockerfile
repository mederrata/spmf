FROM python:3.8-slim-buster

ENV PYTHONPATH /code

RUN apt-get -y update
RUN apt-get -y upgrade
RUN mkdir /code
ADD . /code/

RUN python3 -m pip install /code/
