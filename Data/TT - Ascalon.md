[[Atlantis]] is the development SoC using Ascalon cluster. ASC silicon prof of concept. 

![[Pasted image 20250513160941.png]]
# Modern Out-of-Order Microprocessor Pipeline: High-Level Block Diagram Explanation
---
## 1. Instruction Fetch Stage (Top Section)

### Components:
- **Branch Prediction Complex:** Predicts branch outcomes to fetch correct next instructions and avoid stalls.
  - Includes: TAGE, BDP, BTP, RAS (various branch predictors and return address stack)
  - **Branch State Table:** Stores recent branch history for prediction.
- **Instruction [[Cache]] Complex:** Fast memory storing recently used instructions.
  - **ICACHE:** Instruction cache itself.
  - **ITLB:** Instruction Translation Lookaside Buffer for virtual-to-physical address translation.
  - **IC WAY-PREDICTOR:** Predicts cache way to reduce access time.
  - **ICTAG:** Stores cache line tags for hit/miss detection.
  - **PREFETCH:** Fetches likely-needed instructions early.
- **32B Fetch:** Fetches instructions in 32-byte chunks.
- **Fetch Buffer:** Temporarily holds fetched instructions.
- **Aligner:** Aligns instructions spanning multiple fetch blocks and outputs micro-operations (Uops).

### How it Works:
1. Branch Prediction Complex predicts next instruction address.
2. Instruction Cache accessed using predicted physical address from ITLB.
3. On cache hit, instructions retrieved; on miss, request sent to lower memory.
4. Instructions fetched in 32-byte chunks placed in Fetch Buffer.
5. Aligner breaks raw bytes into instructions and micro-operations (Uop0 to Uop7).

---

## 2. Decode and Rename Stage (Middle Section)

### Components:
- **DEC 0-7:** Multiple parallel decoders translating instructions into micro-operations (Uops).
- **Mapper:** Performs register renaming, replacing architectural registers with physical registers to avoid false dependencies.
  - Uses **INT PR FREELIST** and **FP PR FREELIST** to allocate free physical registers.
- **Checkpoints:** Store processor state for speculative execution rollback.
- **History File:** Tracks rename history for precise exception handling.
- **Retire Queue:** Holds executed instructions awaiting retirement in program order.

### How it Works:
1. Decoders convert aligned instructions into one or more Uops.
2. Mapper allocates physical registers and updates mapping tables.
3. Rename info stored in History File and Retire Queue; checkpoints taken before speculative branches.

---

## 3. Execution Stage (Bottom Section)

### Components:
- **Dispatch Buffers (DB0-DB4):** Hold renamed Uops waiting to be issued.
- **Reservation Stations (RS 0-7 per DB):** Hold ready Uops and monitor operands.
- **Integer Register File (INT RF):** Stores integer physical register values.
- **Floating-Point Register File (VFP RF):** Stores floating-point physical register values.
- **Integer Unit:** Multiple ALUs for integer operations (ALU0, ALU1, ALU2, ALU SHFT, ALU MUL, ALU DIV).
- **Floating-Point Unit (VFP UNIT):** Units for floating-point operations (VFP0, VFP1, VFP2).
- **Load/Store Unit:**
  - **DC Way-Predictor:** Predicts data cache way.
  - **DTLB:** Data Translation Lookaside Buffer.
  - **DCACHE (128KB x 8W):** Data cache.
  - **DCTAG:** Data cache tags.
  - **LDQ (Load Queue):** Holds pending load operations.
  - **STQ (Store Queue):** Holds pending store operations.
  - **Pattern/Stride Prefetcher:** Predicts future data accesses.
  - **Address Queue:** Holds load/store addresses.
  - **Write Data Queue:** Buffers data to be written to memory.
  - **Fill Buffer:** Temporarily stores data fetched due to cache misses.

### How it Works:
1. Uops with ready operands in reservation stations are dispatched to execution units.
2. Execution units perform operations; results written back to physical registers and broadcast.
3. Load/Store Unit calculates addresses, accesses DCACHE or fetches from memory on miss.
4. Store operations buffered before writing to cache and memory.

---

## 4. Retire Stage (Back to Middle Section)

### Components:
- **Retire Queue:** Holds executed instructions to retire in program order.

### How it Works:
1. Instructions retired in original fetch order.
2. Results committed to architectural register file.
3. Memory updates from stores become visible.
4. Physical registers freed and returned to free lists.

---

## Overall Pipeline Flow

1. **Fetch:** Instructions fetched using branch prediction.
2. **Decode & Rename:** Instructions decoded into Uops; registers renamed.
3. **Dispatch:** Renamed Uops buffered awaiting operands.
4. **Execute:** Ready Uops executed out-of-order.
5. **Writeback:** Results written to physical registers.
6. **Retire:** Instructions retired in program order, making changes permanent.

---

## Key Concepts Illustrated

- **Out-of-Order Execution:** Executes instructions as soon as operands are ready, not strictly in program order.
- **Speculative Execution:** Predicts branch outcomes and executes ahead; rollbacks on misprediction using checkpoints.
- **Register Renaming:** Avoids false data dependencies by mapping architectural registers to physical ones.
- **Caches:** Instruction and data caches reduce memory access latency.
- **Translation Lookaside Buffers (ITLB and DTLB):** Accelerate virtual-to-physical address translation.
- **Pipelining:** Overlaps instruction processing stages to increase throughput.

---

This diagram and explanation showcase the complexity and sophistication of modern high-performance microprocessors, which leverage these techniques to maximize instruction-level parallelism and performance.
