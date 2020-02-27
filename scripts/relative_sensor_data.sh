#!/bin/bash

# Download relative pressure sensor data

#!/bin/bash
if [ "$#" -ne 5 ]; then
	echo -e "Usage:\n\thigh_freq_data.sh <sensor_id> <start_timestamp> <end_timestamp> <environment> <destination_file>"
	exit 0
fi

sensor_id=$1
start_time=$2
end_time=$3
environment=$4
destination=$5

args="\"$sensor_id\" \"$start_time\" \"$end_time\" tmp_file.npz \"$environment\" \"single-npz\""
cmd="lein audio $args"

cd "legend-live"
echo $cmd
eval $cmd

cd ..
mv "legend-live/tmp_file.npz" $destination
