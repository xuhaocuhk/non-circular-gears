import importlib

drive_gear_name = "circular_gear"
number_of_samples = 4096

if __name__ == '__main__':
    drive_gear_module = importlib.import_module(f"drive_gears.{drive_gear_name}")
    assert hasattr(drive_gear_module, "generate_gear")
    angle_samples = drive_gear_module.generate_gear(number_of_samples)

    
