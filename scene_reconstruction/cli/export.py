"""Commands for data export."""


import typer
from hydra.utils import instantiate

from scene_reconstruction.cli.config import make_cfg
from scene_reconstruction.data.nuscenes.scene_flow import SceneFlow
from scene_reconstruction.occupancy.temporal_transmission_and_reflection import TemporalTransmissionAndReflection
from scene_reconstruction.occupancy.transmission_reflection import ReflectionTransmissionSpherical

app = typer.Typer(name="export", callback=make_cfg, help="Various export commands.", no_args_is_help=True)

SAVE_DIR = typer.Option(help="Directory to save data to.", dir_okay=True)
BATCH_SIZE = typer.Option(help="Batch size for data processing.")


# pixi run transmissions-reflections
@app.command(name="transmissions-reflections")
def transmissions_reflections(ctx: typer.Context) -> None:
    """Export sensor count maps to specified path."""
    cfg = ctx.meta["cfg"]

    transmission_and_reflections: ReflectionTransmissionSpherical = instantiate(cfg.export.transmissions_reflections)
    transmission_and_reflections.process_data()

    # default.yaml export:
    #   transmissions_reflections:
    #     _target_: scene_reconstruction.occupancy.transmission_reflection.ReflectionTransmissionSpherical
    #     ds:
    #       _target_: scene_reconstruction.data.nuscenes.dataset.NuscenesDataset
    #       data_root: data/nuscenes
    #       extra_data_root: data/nuscenes_extra
    #       version: v1.0-mini
    #       key_frames_only: false

    #     extra_data_root: data/nuscenes_extra
    #     spherical_lower: [2.5, 1.3089969389957472, -3.141592653589793] #[0.0, (90 - 15) / 180 * math.pi, -math.pi]
    #     spherical_upper: [60.0, 2.1816615649929116, 3.141592653589793] #[0.0, (90 + 35) / 180 * math.pi, -math.pi]
    #     spherical_shape: [600, 100, 720] # voxel size [0.1m, 0.5°, 0.5°]
    #     lidar_min_distance: 2.5 # meters
    #     # voxel_size cartesian = 0.2m
    #     # voxel_size sperical : 0.1m, 0.5°, 0.5°
    #     batch_size: 4

REF_KEYFRAME_ONLY = typer.Option(help="Only accumulate for reference keyframes.")


@app.command(name="temporal-accumulation")
def temporal_transmissions_reflections(
    ctx: typer.Context,
) -> None:
    """Accumulate sensor count maps over time to specified path."""
    cfg = ctx.meta["cfg"]

    temporal_accumulation: TemporalTransmissionAndReflection = instantiate(cfg.export.temporal_accumulation)
    temporal_accumulation.process_data()


@app.command(name="scene-flow")
def scene_flow(ctx: typer.Context) -> None:
    """Accumulate sensor count maps over time to specified path."""
    cfg = ctx.meta["cfg"]

    scene_flow: SceneFlow = instantiate(cfg.export.scene_flow)
    scene_flow.process_data()


@app.command(name="sensor-belief-maps", no_args_is_help=True)
def sensor_belief_maps(ctx: typer.Context) -> None:
    """Export sensor belief maps to specified path."""
