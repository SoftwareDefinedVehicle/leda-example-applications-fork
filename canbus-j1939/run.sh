#!/bin/bash

python3 ./dbcinfo.py \
    -I vehicle_signal_specification/spec \
    -u vehicle_signal_specification/spec/units.yaml \
    --extended-attributes dbc \
    --vspec-file vehicle_signal_specification/spec/VehicleSignalSpecification.vspec \
    --dbc-file j1939.dbc \
    --unit-file vehicle_signal_specification/spec/units.yaml \
    --output-vspec-file custom.vspec

echo "Converting custom.vspec to custom_vss_dbc.json..."
vspec2json.py \
    -e "dbc" \
    -o custom.vspec \
    --json-pretty \
    --json-all-extended-attributes \
    --unit-file vehicle_signal_specification/spec/units.yaml \
    vehicle_signal_specification/spec/VehicleSignalSpecification.vspec \
    custom_vss_dbc.json

docker compose up -d databroker

timeout 15s python3 ./kuksa.val.feeders/dbc2val/dbcfeeder.py --config dockerfiles/dbc_feeder_local.ini 2> dbc_feeder.log
