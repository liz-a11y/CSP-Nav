# Private configuration location

The public release intentionally omits `config.py`, `config_mappo.py`, and all experiment-specific configuration files.

For local training, place the two private files in this directory. The repository `.gitignore` excludes them. The MAPPO module must expose a `Config` class and may import the base configuration from `crowd_nav.configs.config`.
