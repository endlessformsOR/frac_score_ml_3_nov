FROM python:3.6.9

RUN apt-get update && apt-get upgrade -y

#Install Poetry
RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python

#Install Vault
RUN wget https://releases.hashicorp.com/vault/1.4.2/vault_1.4.2_linux_amd64.zip -O vault.zip && \
    unzip vault.zip && mv vault /usr/bin

RUN mkdir /legend-analytics

COPY pyproject.toml  /legend-analytics/
COPY model_runner/model_runner.py /legend-analytics/
ADD utils /legend-analytics/utils

WORKDIR /legend-analytics

#Upgrade pip (need to use poetry here for poetry install to work)
RUN /root/.poetry/bin/poetry run pip install --upgrade pip

RUN /root/.poetry/bin/poetry install
