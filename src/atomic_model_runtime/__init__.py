"""Public installed runtime for restricted, explicitly experimental EOS models."""


def create_eos_episode(config=None):
    from atomic_model_runtime.episode import EOSEpisode

    return EOSEpisode(config)


__all__ = ["create_eos_episode", "create_eos_assembly"]


def create_eos_assembly(spec, *, max_step_s=0.0005):
    from scripts.controlled_eos_assembly import ControlledEOSAssembly

    return ControlledEOSAssembly(spec, max_step_s=max_step_s)
