#!/bin/bash
script_dir=$(dirname $0)
source $script_dir/instance-id.sh
source $script_dir/legend-access
aws ec2 stop-instances --instance-ids $instance_id
