#!/bin/bash
sudo apt update

# Install Vault
sudo apt install unzip
wget -q https://releases.hashicorp.com/vault/0.10.1/vault_0.10.1_linux_amd64.zip -O vault.zip
unzip vault.zip
sudo mv vault /usr/bin/vault
sudo ln /root/.vault-token ~/.vault-token
vault auth $VAULT_TOKEN

# Install Python
sudo apt install python3-pip
pip3 install pipenv
export PATH="${HOME}/.local/bin:$PATH"
