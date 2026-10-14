# Optional tactile simulation

[中文](tactile_zh.md) · [Installation](tactile-installation.md) · [Three-task demo](../tests/tactile/integration/README.md)

The normal RoboDojo entry point accepts `--env-cfg arx_x5_pad_tactile`.
The stock `arx_x5` preset keeps tactile simulation disabled.

The CAD preset uses the original dual-X5 layouts. Two 1 mm pads are attached to
**the two fingers of the left gripper**; the right arm is unchanged.
The additional asset directory is `Assets/Robots/x5_pad`, which still references
`Assets/Robots/x5`. Keep both directories and the original CAD source.

## Run

Activate the installed simulation environment. Start a policy server, then use:

```bash
bash scripts/robodojo.sh client \
  --task plug_in_charger --env-cfg arx_x5_pad_tactile \
  --policy-name demo_policy --policy-host 127.0.0.1 --policy-port 19091 \
  --eval-num 1 --action-type ee --num-envs 1 --seed 0 --gui
```

For the supplied contact controller, use the complete paired commands in the
[three-task demo instructions](../tests/tactile/integration/README.md).
Omit `--gui` to run headless. `--eval-num` counts episodes; `--num-envs` counts
parallel environments. The current tactile implementation supports one environment.

## Observations and video

Tactile sensors follow normal scene initialization, physics steps, observation
capture, reset and close. `obs["tactile"]["left" / "right"]` contains RGB,
reference RGB, validity and frame timing. These names identify the two fingers.
The evaluator writes normal camera videos and a `*_visuotactile_*.mp4` mosaic
showing three cameras and both tactile images. It does not automatically save NPZ.

The accepted preset uses UIPC contact/FEM physics with `optical_input: object`:
virtual object depth is mapped through the GelSight Mini Taxim response.
`deformed_surface` remains an optional optical input, but is not used by this preset.
The preset, source CAD geometry and solver parameters define the reproduced demo;
changing optical input is a separate experiment.

## Assets and dependencies

[Installation](tactile-installation.md) describes the pinned UniVTAC/TacEx/libuipc
sources and checks. To regenerate the CAD assets, use
`scripts/internal/import_x5_pad.py --source-dir <unpacked-CAD-directory>` in that
environment. The generated source manifest records the input files and hashes.
Do not reimport assets just to run an existing installation.
