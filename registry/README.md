# NeMA Registry

This folder is a non-intrusive registry/index for the current analysis and visualization pipeline.
It does not modify any existing scripts or outputs.

## Purpose

- Track which script produces which artifact.
- Keep metric definitions in one place.
- Provide a stable starting point for a future HTML dashboard.

## Files

- `pipeline_registry.csv`: master map of script -> input -> output -> metric.
- `artifact_inventory.csv`: curated inventory of key CSV/PNG artifacts currently in use.
- `metric_dictionary.csv`: concise metric definitions and intended usage.

## Usage

- Update registry files here when adding new scripts or outputs.
- Keep paths absolute for reproducibility.
- Treat this folder as documentation + metadata only (no logic dependencies).
