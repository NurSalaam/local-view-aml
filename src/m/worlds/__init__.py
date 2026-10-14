"""World loaders: raw released data to one canonical payments table per world.

Each loader owns its world's conventions (padded
identifiers, node keys, split rules, pseudo-bank filters), so every census
and detector reads the same table. Caches under ``data/interim`` are derived
projections: delete them and the loader rebuilds from raw.
"""
