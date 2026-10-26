"""Короткий прогон номинала сосредоточенной установки ВВЭР."""

from plants.vver import VVERPlant


def main():
    plant = VVERPlant()
    before = plant.observation()
    after = plant.advance(0.5, rtol=1e-4, atol=1e-2)
    print("passport", plant.passport.model_id, plant.passport.status.value)
    print("plant_referenced", plant.passport.plant_referenced)
    print("thermal_MW", after["thermal_power_w"] / 1e6)
    print("electric_MW", after["electrical_power_w"] / 1e6)
    print("flow_kg_s", after["mass_flow_kg_s"])
    print("sg_MPa", after["sg_pressure_pa"] / 1e6)
    print(
        "drift_ppm",
        (after["thermal_power_w"] - before["thermal_power_w"]) / before["thermal_power_w"] * 1e6,
    )


if __name__ == "__main__":
    main()
