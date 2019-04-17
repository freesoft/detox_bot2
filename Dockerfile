FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive
ENV PYSPARK_PYTHON=python3
ENV TERM xterm
ENV LC_ALL C.UTF-8
ENV FLASK_APP webapp.py
LABEL maintainer="Cindy Tseng(cindyst2@illinois.edu)"

# install java
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential\
        expect git vim zip unzip wget openjdk-8-jdk wget sudo

# install python3
RUN apt-get install -y python3 python3-pip python3-dev 

# Download and install spark
RUN     cd /usr/local/ &&\
    wget "http://apache.cs.utah.edu/spark/spark-2.4.1/spark-2.4.1-bin-hadoop2.7.tgz" &&\
        tar -xvzf spark-2.4.1-bin-hadoop2.7.tgz && \
        ln -s ./spark-2.4.1-bin-hadoop2.7 spark &&  \
        rm -rf /usr/local/spark-2.4.1-bin-hadoop2.7.tgz && \
        rm -rf /usr/local/spark/external && \
        chmod a+rwx -R /usr/local/spark/

# download requirements
COPY requirements.txt /requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r /requirements.txt && \
    pip3 freeze
    
RUN echo "alias spark-submit='/usr/local/spark/bin/spark-submit'" >> ~/.bashrc

# Ensure spark log output is redirected to stderr
RUN cp /usr/local/spark/conf/log4j.properties.template /usr/local/spark/conf/log4j.properties

# Set relevant environment variables to simplify usage of spark
ENV SPARK_HOME /usr/local/spark
ENV PATH="/usr/local/spark/bin:${PATH}"
RUN chmod a+rwx -R /usr/local/spark/

# setup flask
RUN mkdir -p /app
EXPOSE 5000

COPY . /app
WORKDIR /app

# prepare classifier and vectorizer so webapp will start up faster
RUN python3 ./detox_engine.py

ENTRYPOINT ["python3", "-u"]
CMD ["/app/webapp.py"]
