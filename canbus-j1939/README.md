# Eclipse Leda - Example for CAN J1939

The goal of this example use case is to show

- building of custom Kuksa DBC Feeder containers as **Vehicle Service** implementation
- showing access to CAN J1939 signals via Vehicle Signals Specification **for Vehicle Applications**

## Architecture Overview

- dbc2val container, configurable with new custom DBC + Mapping files
- mock for providing simulated J1939 signals on CAN-Bus
- Velocitas app accessing J1939 information and providing it as a service to remote cloud services

## DBC and VSS Mapping

DBC files are custom and VSS is standardized, there needs to be some kind of mapping between both worlds.
The mapping is currently done manually using kuksa.val dbc2val mapping files, which get integrated into the merged json file.

<https://github.com/nberlette/canbus> is providing pre-built universal DBC files for J1939 and OBD2.

After modifying custom.vspec and custom.dbc, regenerate the resulting VSS model:

```shell
docker run --name vspec2json --rm -v `pwd`:/data ghcr.io/eclipse-leda/leda-vss-vspec2json:main
```

## Auto-generated mapping

For quick experiments, it may be easier to auto-generate a mapping.
Use the `run.sh` script to (re-)generate the mapping.

> *Attention:* The generated mappings are **proposals only**. They may match on name and unit/datatype,
but there is only a small chance that they actually match semantically! Do NOT use the generated mappings
for actual vehicle signals. **This is dummy data for simulation, testing and mocking purposes only.**

## Databroker CLI

```shell
docker run it --rm ghcr.io/eclipse/kuksa.val/databroker-cli:master
```

### References

- <https://eclipse-leda.github.io/leda/docs/app-deployment/kuksa-databroker/custom-vss-mappings/>
- <https://github.com/eclipse/kuksa.val.feeders/blob/main/dbc2val/mapping/mapping.md>
- <https://github.com/eclipse/kuksa.val.feeders/tree/main/dbc2val>