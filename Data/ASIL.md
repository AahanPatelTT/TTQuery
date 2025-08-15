> ASIL refers to **Automotive Safety Integrity Level**. It is a risk classification system defined by the ISO 26262 standard for the functional safety of road vehicles.

ASILs establish safety requirements―based on the probability and acceptability of harm―for automotive components to be compliant with [ISO 26262](https://www.iso.org/search.html?PROD_isoorg_en%5Bquery%5D=ISO26262).

There are four ASILs identified by ISO 26262―A, B, C, and D. ASIL **A represents the lowest degree and ASIL D represents the highest degree of automotive hazard**

Systems like airbags, anti-lock brakes, and power steering require an ASIL-D grade―the highest rigor applied to safety assurance―because the risks associated with their failure are the highest. On the other end of the safety spectrum, components like rear lights require only an ASIL-A grade. Head lights and brake lights generally would be ASIL-B while cruise control would generally be ASIL-C.

![](https://images.synopsys.com/is/image/synopsys/asil-classifications?qlt=82&wid=1200&ts=1716308461286&$responsive$&fit=constrain&dpr=off)
## How Do ASILs Work?

ASILs are established by performing hazard analysis and risk assessment. For each electronic component in a vehicle, engineers measure three specific variables:

- Severity (the type of injuries to the driver and passengers)
- Exposure (how often the vehicle is exposed to the hazard)
- Controllability (how much the driver can do to prevent the injury)

Each of these variables is broken down into sub-classes. Severity has four classes ranging from “no injuries” (S0) to “life-threatening/fatal injuries” (S3). Exposure has five classes covering the “incredibly unlikely” (E0) to the “highly probable” (E4). Controllability has four classes ranging from “controllable in general” (C0) to “uncontrollable” (C3).

All variables and sub-classifications are analyzed and combined to determine the required ASIL. For example, a combination of the highest hazards (S3 + E4 + C3) would result in an ASIL D classification.