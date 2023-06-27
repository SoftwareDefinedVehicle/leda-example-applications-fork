#!/bin/bash

python3 ./dbcinfo.py \
    -I vehicle_signal_specification/spec \
    -u vehicle_signal_specification/spec/units.yaml \
    -o custom.vspec \
    --extended-attributes dbc \
    vehicle_signal_specification/spec/VehicleSignalSpecification.vspec 
