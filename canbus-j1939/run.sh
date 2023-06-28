#!/bin/bash

python3 ./dbcinfo.py \
    -I vehicle_signal_specification/spec \
    -u vehicle_signal_specification/spec/units.yaml \
    -o custom.vspec \
    --extended-attributes dbc \
    --vspec-file vehicle_signal_specification/spec/VehicleSignalSpecification.vspec \
    --dbc-file j1939.dbc \
    --output-vspec-file out-mapping.vspec
