#!/bin/bash
env=dev
VAULT_ADDR='https://vault.originrose.com:8200'

db_creds_path=secret/legend/sql/legend-${env}

db_name=$(vault read -field=dbname ${db_creds_path})
db_host=$(vault read -field=host ${db_creds_path})
db_password=$(vault read -field=password ${db_creds_path})
db_port=$(vault read -field=port ${db_creds_path})
db_user=$(vault read -field=user ${db_creds_path})

aws_creds=`vault read secret/legend/aws/core`
access_key=$(echo "$aws_creds" | awk '/access_key/ { print $2 }')
secret_key=$(echo "$aws_creds" | awk '/secret_key/ { print $2 }')

AWS_ACCESS_KEY_ID=$access_key
AWS_SECRET_ACCESS_KEY=$secret_key
AWS_SESSION_TOKEN=""


docker run -d --restart=always                                    \
       -v /data:/srv/data                                         \
       -e VAULT_ADDR="https://vault.originrose.com:8200"          \
       -e LEGEND_ENV=$env                                         \
       -e db_host=$db_host                                        \
       -e db_name=$db_name                                        \
       -e db_user=$db_user                                        \
       -e db_password=$db_password                                \
       -e db_port=$db_port                                        \
       -e AWS_ACCESS_KEY_ID=$access_key                           \
       -e AWS_SECRET_ACCESS_KEY=$secret_key                       \
       --name model-runner-$env                                   \
       --network="host"                                           \
       docker.originrose.com/legend-model-runner:latest           \
       /root/.poetry/bin/poetry run python model_runner.py
