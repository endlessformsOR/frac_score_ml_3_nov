#!/bin/bash
err_report() {
  echo "errexit on line $(caller)" >&2
}

trap err_report ERR
tmpfile=$(mktemp)
vault read -field=key secret/legend/aws/analytics-ssh > ${tmpfile}
ssh -i "$tmpfile" ubuntu@ec2-44-231-77-148.us-west-2.compute.amazonaws.com
rm ${tmpfile}
