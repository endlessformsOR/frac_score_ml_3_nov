# Analytics

Jupyter Lab running on an R5.2xlarge instance at http://44.231.77.148:9998


## Credentials

* Jupyter Lab: `vault read secret/legend/aws/analytics-jupyter`
* SSH: `vault read secret/legend/aws/analytics-ssh`

## Vault

Vault is necessary for most scripts to work. Vault token on instance is installed with `scripts/token.sh install`.

## Usage

### Start/Stop instance:
- `scripts/stop-instance.sh`
- `scripts/start-instance.sh`

### SSH
`scripts/ssh.sh`

### Download/Upload data

Scripts to sync data
-  Download: `scripts/download-s3-data`
-  Upload: `scripts/upload-s3-data`

## Instance setup

Most of the instance setup I documented in `scripts/instance-setup.sh`. What isn't included is setting up Github (creds for `thinktopic-thinkbot` in `secret/thinktopic/thinktopic-thinkbot`) and setting up a cronjob to run jupyter lab (`cd /home/ubuntu/legend-analytics && pipenv run jupyter lab`).
