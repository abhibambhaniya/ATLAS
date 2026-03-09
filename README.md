# ATLAS: Architecture Toolkit for LLM Acceleration with Sparsity

**ATLAS** is an end-to-end codesign methodology and modeling framework that holistically evaluates compute, memory, and interconnect subsystems to estimate the area, energy, and performance of sparse AI accelerators.

As large language models (LLMs) scale to hundreds of billions of parameters, sparsity has emerged as a crucial technique to reduce computational and memory overheads. However, ignoring interconnect and memory subsystem costs can nullify the potential benefits of sparsity, especially as accelerator scale increases. ATLAS bridges this gap by systematically analyzing tradeoffs between area, energy, and performance under quality constraints across a broad range of modern AI models.

## Key Features

- **Holistic Subsystem Modeling:** Evaluates the sparsity-aware enhancements needed across compute processing elements (PEs), on-chip SRAM, and on-chip interconnects.

- **Diverse Sparsity Support:** Models various sparsity patterns, from N:M structured sparsity (e.g., 2:4, 1:8) to fine-grained unstructured sparsity.

- **Accurate Area & Energy Estimation:** Integrates with CACTI 7 to maximize SRAM memory density and DSENT for interconnect crossbar area modeling, scaling all reported areas to a 7-nm process node.

- **Design Space Exploration:** Performs systematic architectural searches to identify Pareto-efficient hardware configurations tailored to specific deployment contexts, including cloud-scale training, datacenter inference, and edge inference.

## Supported Compute Architectures

ATLAS evaluates compute cores using different PE architectures depending on the target sparsity pattern:

- **Dense Baseline:** Modeled using a systolic array architecture.

- **Structured N:M:** Implements a VEGETA-like PE supporting variable $N \in \{1, 2, 4\}$ when $M=4$, providing up to 4× compute compression versus dense compute.

- **Unstructured:** Modeled based on SIGMA, incorporating a configurable distribution and reduction network to handle irregular dataflows efficiently.

## Optimization Objective

The framework formalizes hardware design as a constrained optimization task. It evaluates designs based on a Total Cost of Ownership (TCO) merit function that balances performance against manufacturing (area) and operational (energy) costs:

$$merit(s, w) = \frac{performance(s, w)}{area(s) + 4 \cdot energy(s, w)}$$

## Getting Started

```bash
# Clone the repository
git clone https://github.com/abhibambhaniya/ATLAS.git
cd ATLAS

# Install uv (if not already installed)
pip install uv

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Usage Setup

Run the ATLAS CLI tool to estimate latency, area, and energy for a given model and hardware configuration.

```bash
# 1. Basic run – structured 2:4 sparsity on H100
python run_atlas.py --model llama3-8b --sparsity_type N:M --ratio 2:4 --hardware h100

# 2. Different model
python run_atlas.py --model gemma2-2b --sparsity_type N:M --ratio 1:4 --hardware a100

# 3. Unstructured sparsity
python run_atlas.py --model llama3-8b --sparsity_type unstructured --hardware h100

# 4. With optional overrides
python run_atlas.py --model llama3-8b --sparsity_type N:M --ratio 2:4 --hardware h100 --batch_size 32 --isl 512 --osl 128

# 5. Help text
python run_atlas.py --help
```

## Citation

If you use ATLAS in your research, please cite our paper:

```
@ARTICLE{11410568,
  author={Bambhaniya, Abhimanyu R. and Kao, Sheng-Chun and Jeong, Geonhwa and Subrimaniam, Suvinay and Yazdanbaksh, Amir and Krishna, Tushar},
  journal={IEEE Micro},
  title={Demystifying the cost versus benefits of sparse LLM acceleration},
  year={2026},
  volume={},
  number={},
  pages={1-12},
  keywords={Computational modeling;Hardware;Accuracy;Costs;Artificial intelligence;Runtime;Memory management;Tensors;Decoding;Random access memory},
  doi={10.1109/MM.2026.3667115}}
```

## License

This project is licensed under the MIT License.
