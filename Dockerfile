FROM python:3.6.9

RUN apt-get update && apt-get upgrade -y

#Install Poetry
RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python


#Install Vault
RUN wget https://releases.hashicorp.com/vault/1.4.2/vault_1.4.2_linux_amd64.zip -O vault.zip && \
    unzip vault.zip && mv vault /usr/bin


RUN mkdir /model_runner
COPY pyproject.toml db.py dynamic_utils.py model_runner.py /model_runner/

WORKDIR /model_runner
RUN /root/.poetry/bin/poetry install
