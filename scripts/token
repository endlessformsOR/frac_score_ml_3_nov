#!/bin/bash
# This script runs on production server to renew vault token periodically. The
# script may only be run by an administrator which has access to the core role.
set -e
script_dir=$(dirname $0)
script_name=$(basename $0)
err_report() {
  echo "errexit on line $(caller)" >&2
}

trap err_report ERR
if [ "$1" == "install" ]; then

    role_id=$(vault read --field=role_id auth/approle/role/legend/role-id)
    secret_id=$(vault write --field=secret_id -f auth/approle/role/legend/secret-id)
    vault_token=$(vault write --field=token auth/approle/login role_id=$role_id secret_id=$secret_id)
    source $script_dir/getkey.sh
    scp -i "$credfile" $0 "$user@$host:"
    ssh -i "$credfile" $user@$host << EOF
        sudo sh -c "export VAULT_ADDR='https://vault.originrose.com:8200'; echo -n \"$vault_token\" > /root/.vault-token"
        sudo mv $script_name /etc/cron.hourly
        sudo chown root:root /etc/cron.hourly/$script_name
EOF
    exit 0
fi

export VAULT_ADDR='https://vault.originrose.com:8200'
vault token renew
