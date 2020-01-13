#!/bin/bash
script_dir=$(dirname $0)

err_report() {
  echo "errexit on line $(caller)" >&2
}

trap err_report ERR
source $script_dir/sshkey.sh
ssh -i "$credfile" $user@$host
rm ${credfile}
