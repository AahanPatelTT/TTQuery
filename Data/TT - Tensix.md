Tensix Intellectual Property (IP) represents Tenstorrent's family of proprietary, high-performance Artificial Intelligence (AI) accelerator cores. These cores are meticulously engineered for a wide spectrum of machine learning (ML) workloads, encompassing both complex training and efficient inference tasks.

While these RISC-V cores handle general control and dispatch, the primary AI computation is executed by specialized, powerful math and SIMD (Single Instruction, Multiple Data) engines optimized for tensor operations. This architectural choice leverages the inherent flexibility and openness of the RISC-V standard, while simultaneously allowing Tenstorrent to implement highly specialized and optimized hardware for AI acceleration.

[Tensix Core](https://tenstorrent.atlassian.net/wiki/x/FIDpAQ "https://tenstorrent.atlassian.net/wiki/x/FIDpAQ")
### Evolution of Tensix Products

| Feature                  | Grayskull (Tensix)    | Wormhole (Tensix+)              | Blackhole (Tensix++)           | Quasar (Tensix NEO)             |
| ------------------------ | --------------------- | ------------------------------- | ------------------------------ | ------------------------------- |
| **Process Node**         | 12nm                  | 12nm                            | 6nm                            | 4nm Chiplet                     |
| **Tensix Cores (Max)**   | 120                   | 80 per ASIC (up to 128 on card) | 140                            | 32 per chiplet                  |
| **SRAM per Core**        | 1MB                   | 1.5MB                           | 1.5MB                          | TBD (Likely >=1.5MB)            |
| **Total SRAM (Example)** | Up to 120MB (e150)    | Up to 192MB (n300s)             | Up to 210MB (p150)             | TBD (Scalable with chiplets)    |
| **Peak FP8 TFLOPS**      | ~332 (e150)           | ~466 (n300s)                    | ~774 / ~1 POPS (INT8)          | TBD (Scalable with chiplets)    |
| **Peak BLOCKFP8 TFLOPS** | ~83 (e150)            | ~262 (n300s)                    | ~387                           | TBD                             |
| **Memory Type**          | LPDDR4                | GDDR6                           | GDDR6                          | TBD (Chiplet dependent)         |
| **Host CPU Integration** | External              | External                        | Integrated (16x RISC-V X280)   | Integrated/Chiplet option       |
| **Chip-to-Chip Scale**   | PCIe only             | 100Gbps Ethernet NoC extension  | 400Gbps Ethernet NoC extension | Non-blocking D2D interfaces     |
| **Key Innovation**       | Baseline Tensix Arch. | Scalable Ethernet, More SRAM    | On-chip CPUs, 6nm process      | Low-power AI Chiplet, Stackable |
| **Licensable IP Gen.**   | (Internal/Baseline)   | (Internal/Evolution)            | (Internal/Evolution)           | Tensix NEO                      |

## Overview of Tensix-powered Chips

*   **Grayskull (Gen1 - [[Tensix cores]]):** Tenstorrent's first-generation AI processor, primarily targeting inference workloads and establishing the foundational Tensix architecture. It features up to 120 Tensix cores on a 12nm process, offering approximately 332 [[TFLOPS]] (FP8) and is equipped with 8GB of LPDDR4 memory. Software support has been discontinued. **POC**

*   **Wormhole (Gen2 - Tensix+ cores):** Introduced enhanced "Tensix+" cores, focusing on improved scalability and broader workload applicability, including training. Each Wormhole ASIC contains 80 Tensix+ cores on a 12nm process, with increased SRAM per core (1.5MB) and enhanced BLOCKFP8 performance. Key innovation is the integration of high-speed Ethernet connectivity (16x100 Gbps), enabling direct chip-to-chip scaling.**POC**

*   **Blackhole (Gen3 - Tensix++ cores):** Represents a major leap in performance and integration, positioning it as a standalone AI computer. It features 140 "Tensix++" cores on a 6nm process and employs 8 channels of GDDR6 memory. The most notable innovation is the integration of 16 general-purpose RISC-V CPU cores directly on the chip, along with significantly faster Ethernet connectivity (12x400 Gbps).

## [[Tensix NEO]]: The Licensable AI Core

Tensix NEO is the branding for Tenstorrent's latest generation of licensable AI IP, representing the culmination of architectural advancements. It is optimized for demanding AI/ML workloads, rigorously silicon-tested, offers adaptability and scalability, is power-efficient, and is designed to enable licensees a rapid path to market with their custom SoCs. Tensix NEO is associated with the "[[Quasar]]" low-power AI chiplet, designed with 32 Tensix NEO cores and targeted for fabrication on an advanced 4nm process node. This is engineered for cutting-edge semiconductor processes and is a cornerstone of Tenstorrent's strategy for modular, chiplet-based AI solutions.

