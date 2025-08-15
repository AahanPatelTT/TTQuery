 # Background - Automotive chip market
### 2.1 How is the auto market changing?
- Shift towards [[Software Defined Vehicles]](SDVs) and electrification is increasing software demands on vehicles.
- Mass adoption of higher [[May/ADAS]] levels and EVs in North America, Asia, and Europe is expected.
- Onboard compute requirements for ADAS and EVs continue to scale.

### 2.1.1 Use of Semiconductors in Automotive Industry
- ADAS and infotainment are crucial in modern electrified and semi-autonomous vehicles.
- Automotive CPUs are essential for ADAS, infotainment, and other critical vehicle functions.

### 2.1.1.1 Projections
- By 2030, 58% of light vehicles will have L2 or higher ADAS. ([[May/ADAS#^212c5c|Levels of ADAS]])
- Automotive semiconductor market expected to grow **9% annually, reaching $100B** TAM by 2030.
- Semiconductor consumption per vehicle will rise significantly, with **L5 ADAS EVs requiring ~$2500 in semiconductors vs. $435 for Internal Combustion Engine vehicles in 2020**.

### 2.2 Barriers to Entry
- High entry barriers due to [[Functional safety]] (FuSa) requirements like ISO26262 and [[ASIL]].
- FuSa certification involves millions in investment and extended development time.
- Mature, pre-certified IP can reduce time-to-market by providing safety features and validation.

## 3 Current Market Research
### 3.1 Current Automotive Chip Supply Chain
- Original Engine Manufacturer (e.g., BMW, Honda) set requirements.
- Tier 1 integrators (e.g., Bosch, Denso) develop systems integrating chips and sensors.
- Tier 2 chip designers (e.g., NVIDIA, Qualcomm, Huawei) develop SoCs and IP providers (e.g., ARM, SiFive).
- Tier 3 suppliers provide materials and tooling (e.g., TSMC).

### 3.2 Current CPU Offerings in Automotive Market
- Table of automotive CPU IP offerings shows products from Tenstorrent, ARM, SiFive with ASIL B/D certifications and varying performance metrics.

### 3.3 Automotive CPU IP Offerings

#### 3.3.1 Arm
- Dominates automotive market: >85% of IVI(In Vehicle Infotainment) systems, >50% of ADAS.
- Offers Architecture and Processor IP licenses.
- Key IPs: **A78AE** (ASIL D certified), **A720AE**, **Neoverse V3AE** (targeting various ADAS/IVI applications).
- Focus on balance of performance and power consumption.

#### 3.3.2 SiFive
- Licenses Core IP and DesignShare options.
- Automotive CPUs: S7-AD and X280-A targeting high-performance embedded and vector workloads.
- S7-AD has ISO 21434 certification.

#### 3.3.3 Other [[RISC-V]] Considerations
- Andes Technology targets lower performance automotive applications (ECU, battery management).
- Ventana Microsystems’ Veyron V2 is high performance but unclear on ISO26262 compliance.
- Codasip offers L730 (ASIL D) and L31AS (ASIL B) cores with security features.

### 3.4 Automotive Platform Offerings

| Vendor   | Product   | CPU             | ASIL Level | DMIPS/MHz | DMIPS | Target ADAS Level     | Process Node | Release |
| -------- | --------- | --------------- | ---------- | --------- | ----- | --------------------- | ------------ | ------- |
| Nvidia   | Orin X    | A78AE           | B/D        | 11.36     | 240K  | IVI/ADAS (L2)         | N7           | 2022    |
| Nvidia   | Thor X    | Neoverse V3AE   | B/D        | 18.18     | ~500K | IVI/ADAS (L3+)        | N4           | 2024    |
| Qualcomm | SA8650P   | Kryo (A55)      | B/D        | N/A       | 230K  | IVI/ADAS (L2/L3)      | 4nm          | 2024    |
| Tesla    | HW4       | 20xA72          | N/A        | N/A       | ~237K | IVI/ADAS (L2/L3)      | SF7          | 2023    |
| Renesas  | R-car X5H | A720AE          | B/D        | 13.64     | 1000K | IVI/ADAS (L2++/L3/L4) | N3           | 2024    |
| Mobileye | Eye Ultra | 12x MIPS RISC-V | B/D        | N/A       | N/A   | ADAS (L4)             | 5nm          | 2025    |

- Nvidia’s Drive Orin and Thor platforms target ADAS/IVI with high compute and GPU capabilities.
- Qualcomm offerscompetitive pricing with Snapdragon platforms.
- Mobileye Eye Ultra uses 12 RISC-V cores plus ARM GPU and AI cores.

### 3.4.4 Chinese CPU Offerings

| Vendor           | Product    | CPU Configuration     | ASIL/Safety Level | DMIPS/MHz | [[DMIPS]] | Target Application | Process Node | Release        |
| ---------------- | ---------- | --------------------- | ----------------- | --------- | --------- | ------------------ | ------------ | -------------- |
| Horizon Robotics | Journey 6P | A78-based             | D                 | ~11.4     | 410K      | ADAS/IVI           | 7nm          | 2025 (planned) |
| AutoChips        | AC8025     | 2xA76 + 6xA55         | B                 | N/A       | 60K       | IVI                | N/A          | July 2024      |
| SiEngine         | Dragon     | 4xA76 + 4xA55         | AEC-Q100          | 4.7       | 90K       | IVI                | 7nm          | Dec 2023       |
| UNISOC           | Eagle      | 1xA76 + 3xA76 + 4xA55 | AEC-Q100          | 5.3       | 93K       | IVI                | 6nm          | Mar 2023       |
| Rockchip         | One        | 4xA76 + 4xA55         | B                 | 6.4       | 100K      | IVI                | 8nm          | 2022?          |
| SemiDrive        | A7870      | 6xA55                 | D                 | 3.6       | 100K      | IVI                | 16nm         | April 2021     |
| Black Sesame     | RK3588M    | A55                   | B                 | 2.5       | 60K       | IVI                | 16nm         | April 2021     |

- Chinese offerings mainly use A76 and A55 cores targeting IVI.
- Customers seek alternatives competitive with Nvidia Orin (A78AE) and Arm A720AE.

### 3.4.6 Horizon Robotics
- Journey 6P CPU delivers up to 410K DMIPS, targeting L2++/L3/L4 ADAS.
- Journey 5 is ASIL-B Ready; Journey 6 not yet ASIL certified.

### 3.4.7 SemiDrive
- X9 and V9 SoCs with 6 A55 cores at 2.0GHz, performance 36K-100K DMIPS.
- Likely for lower ADAS levels and infotainment.
- Both lines ASIL-B certified.

### 3.4.8 Black Sesame
- Huashan A1000 and A2000 product lines for ADAS.
- A1000 Pro uses A55 cores at 1.5GHz (~60K DMIPS).
- A2000 announced with 16 A78 cores; benchmarks unknown.

## 4 Tenstorrent Ascalon IP

### 4.1 Technical Specifications
- Baseline D8 auto IP targets 2-2.4GHz frequency, 11.46 DMIPS/MHz.
- Projected 5-10% annual SPEC2006 performance increase.
- D8 auto suitable for general compute, ADAS, and IVI.
- PPA projection at 4nm: 3.2GHz, 18 SPEC2K6/GHz, core area 1.6mm², power ~0.9W dynamic.

### 4.1.1 ARM comparison/Future Projections
- Tenstorrent expects D8++ (2026) to match/exceed Arm IP performance.
- Arm A2025 and A2026 projected SPEC2K6/GHz: ~20-22.
- Tenstorrent D8++ projected >22, D8+++ (2027) >25 SPEC2K6/GHz.

## 5 Ecosystem Considerations
- Beyond performance, power, and cost, ecosystem maturity and support are critical for competitiveness.
---

![[CPU IP Competitive analysis.docx.pdf]]