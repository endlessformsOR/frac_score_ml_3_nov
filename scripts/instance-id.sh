#!/bin/bash
export instance_id=$(vault read -field=instance secret/legend/aws/analytics-ssh)
