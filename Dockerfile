FROM ubuntu:18.04
ENV TERM xterm
ENV LC_ALL C.UTF-8
ENV FLASK_APP webapp.py
LABEL maintainer="Wonhee Jung(wonheej2@illinois.edu)"

RUN apt-get update && apt-get install -y \
    software-properties-common
RUN add-apt-repository universe
RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3.6 \
    python3-pip && \ 
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt && \
    pip freeze

RUN mkdir -p /app
EXPOSE 5000

COPY . /app
WORKDIR /app

RUN python3 -m nltk.downloader popular

# prepare classifier and vectorizer so webapp will start up faster
RUN python3 ./detox_engine.py

ENTRYPOINT ["python3", "-u"]
CMD ["/app/webapp.py"]

