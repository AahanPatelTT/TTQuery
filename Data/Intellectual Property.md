In the context of companies like Tenstorrent, Qualcomm, and ARM, **Intellectual Property (IP)** refers to reusable units of logic, circuits, or layouts that can be licensed or used as building blocks in the design of more complex integrated circuits (ICs), also known as chips or System-on-Chips (SoCs). Types of IP:

| Feature        | Soft IP                                                                | Hard IP                                                        |
| -------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------- |
| Delivery       | RTL code (Verilog, VHDL), gate-level netlist                           | Physical layout (e.g., [[GDS]]II)                              |
| Flexibility    | High; customizable and optimizable                                     | Low; fixed physical design                                     |
| Implementation | Synthesized and implemented by the user                                | Directly placed and routed in the chip layout                  |
| Portability    | High; portable across different technologies                           | Low; specific to a particular process technology               |
| Performance    | Depends on user implementation; less predictable                       | Predictable and often high; optimized for the technology       |
| Area           | Depends on user implementation; less predictable                       | Predictable and often small; optimized for the technology      |
| Examples       | CPU cores, memory controllers, digital interfaces, DSP functions (RTL) | Analog/mixed-signal blocks, high-speed PHYs, embedded memories |

- IP blocks can range from entire processor cores to smaller units like memory controllers, communication interfaces (like USB or PCIe), graphics processing units (GPUs), and specialized accelerators for tasks like AI or video processing.
- Companies can either develop their own IP for use in their products or license their IP to other companies for a fee & royalty.
