#!/bin/bash
credfile=$(mktemp)
vault read -field=key secret/legend/aws/analytics-ssh > ${credfile}

host=$(vault read -field=host secret/legend/aws/analytics-ssh)
user=$(vault read -field=user secret/legend/aws/analytics-ssh)
