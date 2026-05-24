import openvino as ov

core = ov.Core()
print("Available devices:", core.available_devices)

for device in core.available_devices:
    print("\nDevice:", device)
    for prop in ["FULL_DEVICE_NAME", "OPTIMIZATION_CAPABILITIES"]:
        try:
            print(f"{prop}: {core.get_property(device, prop)}")
        except Exception as e:
            print(f"{prop}: unavailable ({e})")
