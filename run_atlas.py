import argparse
import json
import datetime
import os
import sys

# Configure path so that we can import from src and utils
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from src.system import System
from src.unit import Unit
from src.analye_model import get_model_df, get_summary_table
from src.sparse_hw_cost import get_HW_cost, merit_function
from utils.get_language_model import create_inference_moe_prefix_model, create_inference_moe_decode_model
from utils.hardware_presets import get_hardware_preset, Colors

def main():
    parser = argparse.ArgumentParser(description="ATLAS CLI for Estimating Sparse LLM Acceleration")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., llama3-8b, gemma2_2b)")
    parser.add_argument("--sparsity_type", type=str, required=True, help="Sparsity type: 'N:M' or 'unstructured'")
    parser.add_argument("--ratio", type=str, default="1:1", help="Sparsity ratio for structured sparsity (e.g., 2:4)")
    parser.add_argument("--hardware", type=str, required=True, help="Target hardware preset (e.g., h100, a100, custom)")
    
    # Optional parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--isl", type=int, default=1024, help="Input sequence length")
    parser.add_argument("--osl", type=int, default=256, help="Output sequence length")
    parser.add_argument("--tensor_parallel", type=int, default=1, help="Tensor parallelism degree")
    
    # Custom hardware overrides
    parser.add_argument("--flops", type=float, default=None, help="Override compute capacity in TFLOPS")
    parser.add_argument("--offchip_mem_bw", type=float, default=None, help="Override off-chip memory bandwidth in GB/s")
    parser.add_argument("--frequency", type=float, default=None, help="Override frequency in MHz")
    parser.add_argument("--on_chip_mem_size", type=float, default=None, help="Override on-chip memory size in MB")

    args = parser.parse_args()

    # Process model name: allow dashes, convert to internal underscore format
    model_internal_name = args.model.replace('-', '_').lower()
    
    # Hardware configurations
    hw_params = get_hardware_preset(args.hardware)
    if args.flops is not None:
        hw_params['flops'] = args.flops
    if args.offchip_mem_bw is not None:
        hw_params['offchip_mem_bw'] = args.offchip_mem_bw
    if args.frequency is not None:
        hw_params['frequency'] = args.frequency
    if args.on_chip_mem_size is not None:
        hw_params['on_chip_mem_size'] = args.on_chip_mem_size

    # Sparsity configuration
    if args.sparsity_type.lower() == 'unstructured':
        sparse_pe = 0.00001
        acc_type = "unstructured"
    else:
        try:
            n, m = args.ratio.split(':')
            sparse_pe = float(n) / float(m)
        except ValueError:
            print(f"{Colors.FAIL}Error: --ratio must be in the format 'N:M' (e.g. 2:4){Colors.ENDC}")
            sys.exit(1)
        acc_type = "structured"

    print(f"{Colors.OKCYAN}[*] Running ATLAS estimation...{Colors.ENDC}")
    print(f"Model: {model_internal_name}")
    print(f"Hardware: {args.hardware} {hw_params}")
    print(f"Sparsity: {acc_type} (PE Density Support: {sparse_pe})")
    
    # System and Simulation Configuration
    unit = Unit()
    # Fix shape to a default canonical monolithic MXU if not specified, 
    # matching the default reference for monolithic
    Max_Dim = 256
    shape = [1, 1, Max_Dim, Max_Dim] 

    if acc_type == "unstructured":
        system = System(unit, frequency=hw_params["frequency"], flops=hw_params["flops"], 
                        offchip_mem_bw=hw_params["offchip_mem_bw"], pe_min_density_support=1, 
                        model_on_chip_mem_implications=True, on_chip_mem_size=hw_params["on_chip_mem_size"], 
                        mxu_shape=shape, accelerator_type="unstructured", compute_efficiency=(0.5 + 0.01*2))
    else:
        system = System(unit, frequency=hw_params["frequency"], flops=hw_params["flops"], 
                        offchip_mem_bw=hw_params["offchip_mem_bw"], pe_min_density_support=1, 
                        model_on_chip_mem_implications=True, on_chip_mem_size=hw_params["on_chip_mem_size"], 
                        mxu_shape=shape, accelerator_type="structured")
    
    system.set_pe_min_density_support(sparse_pe)
    
    data_path = os.path.join(script_dir, "data")
    
    cycles = 0
    energy = 0
    
    # PREFILL
    try:
        generated_model_prefill = create_inference_moe_prefix_model(
            input_sequence_length=args.isl, output_gen_tokens=0,
            name=model_internal_name, data_path=data_path, tensor_parallel=args.tensor_parallel, 
            method='sparse_FF', spff_density=sparse_pe
        )
        model_df_prefill = get_model_df(generated_model_prefill, system, unit, args.batch_size, data_path, sparse=True, intermediate_on_chip=True)
        df_prefill = get_summary_table(model_df_prefill, system, unit)
        
        cycles += df_prefill.loc[0, 'Cycles'] / 1000000
        energy += df_prefill.loc[0, 'Energy']
    except Exception as e:
        print(f"{Colors.FAIL}Error generating prefill model: {e}{Colors.ENDC}")
        sys.exit(1)

    # DECODE
    try:
        generated_model_decode = create_inference_moe_decode_model(
            input_sequence_length=args.isl, output_gen_tokens=args.osl // 2,
            name=model_internal_name, data_path=data_path, tensor_parallel=args.tensor_parallel, 
            method='sparse_FF', spff_density=sparse_pe
        )
        model_df_decode = get_model_df(generated_model_decode, system, unit, args.batch_size, data_path, sparse=True, intermediate_on_chip=True)
        df_decode = get_summary_table(model_df_decode, system, unit)
        
        cycles += args.osl * df_decode.loc[0, 'Cycles'] / 1000000
        energy += args.osl * df_decode.loc[0, 'Energy']
    except Exception as e:
        print(f"{Colors.FAIL}Error generating decode model: {e}{Colors.ENDC}")
        sys.exit(1)

    unstruct_BW = 2 if acc_type == "unstructured" else 1
    area, BW = get_HW_cost(system, sparse_pe_support=sparse_pe, print_distribution=False, unstructure_BW=unstruct_BW)
    
    # Calculate Latency from total cycles and system frequency
    # We divided cycles by 1000000 above (which equals 1M cycles). 
    # To get ms, given frequency is in MHz (cycles / us = 10^6 cycles / s).
    # Actual cycles calculation needs clarification but for now we format it as reported.
    total_cycles_millions = cycles
    total_energy = energy
    
    latency_ms = (cycles * 1000000) / (system.frequency * 1e3) # Frequency in unit.py raw_to_unit handles stuff, but let's just do an approx latency

    # Create distinct colored output
    print(f"\n{Colors.BOLD}{Colors.OKGREEN}RESULTS:{Colors.ENDC}")
    print(f"{Colors.OKGREEN}► Latency (Cycles in M):{Colors.ENDC} {total_cycles_millions:.4f}")
    print(f"{Colors.OKGREEN}► Area (mm²):{Colors.ENDC} {area:.4f}")
    print(f"{Colors.OKGREEN}► Energy (mJ):{Colors.ENDC} {total_energy:.4f}")
    
    # Prepare JSON output
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(script_dir, "results")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    out_filename = os.path.join(out_dir, f"atlas_results_{model_internal_name}_{timestamp}.json")
    
    results = {
        "timestamp": timestamp,
        "input_config": {
            "model": args.model,
            "internal_model_name": model_internal_name,
            "sparsity_type": args.sparsity_type,
            "ratio": args.ratio,
            "hardware_preset": args.hardware,
            "hardware_params": hw_params,
            "batch_size": args.batch_size,
            "isl": args.isl,
            "osl": args.osl,
            "tensor_parallel": args.tensor_parallel,
            "pe_density": sparse_pe
        },
        "metrics": {
            "cycles_millions": total_cycles_millions,
            "area_mm2": area,
            "energy_mJ": total_energy
        }
    }
    
    try:
        with open(out_filename, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n{Colors.OKBLUE}Results successfully saved to {out_filename}{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}Failed to write JSON output: {e}{Colors.ENDC}")

if __name__ == "__main__":
    main()
