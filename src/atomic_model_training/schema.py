"""Native state order and SI units, bound to the actual core pack layout."""


def state_schema(plant):
    fields = [("neutron_amplitude", "1")]
    fields += [(f"precursor_{i + 1}", "1") for i in range(6)]
    fields += [(f"decay_power_{i + 1}", "W") for i in range(3)]
    fields += [("fuel_temperature", "K"), ("shaft_speed", "rad/s"), ("rod_position", "1")]
    for name in plant.network.volume_names():
        fields += [
            (name + ".mass", "kg"),
            (name + ".internal_energy", "J"),
            (name + ".boron", "kg"),
        ]
    fields += [(name + ".mass_flow", "kg/s") for name in plant.network.junction_names()]
    fields += [("hotwell_mass", "kg"), ("hotwell_energy", "J"), ("sg_wall_temperature", "K")]
    if len(fields) != len(plant.pack()):
        raise ValueError("Core state layout changed")
    return [dict(index=i, name=name, unit=unit) for i, (name, unit) in enumerate(fields)]
