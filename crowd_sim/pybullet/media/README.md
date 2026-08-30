# Runtime assets not included

The PyBullet environment references URDF, OBJ, MTL, DAE and texture files under this directory. Those files were not copied into the public staging folder because their redistribution terms still need to be verified.

Restore the complete dependency chain in the original relative layout before running the simulator; copying only the URDF files is insufficient.
