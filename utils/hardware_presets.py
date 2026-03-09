class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_hardware_preset(hw_name):
    # Lookup table for hardware presets. You can add more here.
    presets = {
        "h100": {"flops": 989, "offchip_mem_bw": 3350, "frequency": 1830, "on_chip_mem_size": 50},
        "a100": {"flops": 312, "offchip_mem_bw": 2039, "frequency": 1410, "on_chip_mem_size": 40},
        "custom": {"flops": 128, "offchip_mem_bw": 5000, "frequency": 1000, "on_chip_mem_size": 128}
    }
    if hw_name.lower() not in presets:
        print(f"{Colors.WARNING}Warning: Hardware '{hw_name}' not found in presets. Using 'custom' preset.{Colors.ENDC}")
        return presets["custom"].copy()
    return presets[hw_name.lower()].copy()

