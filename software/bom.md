# sEMG Hardware Bill of Materials (BOM)

This document outlines the detailed hardware specifications, target part numbers, pricing, and suppliers for the Subvocal neckband physiological silent speech interface.

It covers two hardware options:
1. **$25 Minimum BOM**: A low-cost, off-the-shelf breadboard-ready prototype designed to test basic multi-channel acquisition using standard gel electrodes.
2. **$227 Full BOM**: A custom, battery-powered wearable neckband frontend with a custom PCB, high-CMRR analog instrumentation amplifiers, active analog filters, differential ADC, dry bio-compatible electrodes, and integrated bone conduction audio output.

---

## 1. The $25 Minimum BOM

Designed for rapid bench validation and testing the code stack without custom PCB assembly.

| Component Description | Quantity | Target Part Number | Primary Supplier | Primary Unit Cost | Backup Supplier | Est. Total Cost |
| :--- | :---: | :--- | :--- | :---: | :--- | :---: |
| **ESP32 NodeMCU Development Board** (ESP-WROOM-32, Micro-USB) | 1 | ESP32-DevKitC-32D | DigiKey (356-ESP32-DEVKITC-32D) | $4.80 | Mouser / Amazon | $4.80 |
| **ADS1115 16-Bit 4-Channel I2C ADC Module** (Mini breakout board) | 1 | Adafruit 1085 (ADS1115) | Adafruit (1085) | $5.95 | Amazon / AliExpress | $5.95 |
| **Ag/AgCl Gel Electrode Pads** (Liquid gel, snap connection, pack of 50) | 1 | Covidien Kendall 133 | DigiKey (Covidien 133) | $11.00 | Bio-Medical / Amazon | $11.00 |
| **Snap Electrode Lead Wires** (3.5mm snap to female dupont headers, pack of 10) | 1 | Generic 1m sEMG Leads | Amazon / AliExpress | $3.50 | Adafruit | $3.50 |
| **Solderless Mini Breadboard & Jumper Wires** | 1 | generic-bb-jumpers | Amazon | $2.50 | Adafruit | $2.50 |
| **TOTAL MINIMUM BOM** | | | | | | **$27.75** |

---

## 2. The $227 Full Wearable BOM

Designed for a stand-alone, rechargeable wireless neckband form-factor featuring low-noise differential sEMG amplifiers and bone conduction audio feedback.

### A. Integrated Circuit Components (PCB Assembly)

| Component Description | Qty | Target Part Number | Primary Supplier | Unit Cost | Backup Supplier | Total Cost |
| :--- | :---: | :--- | :--- | :---: | :--- | :---: |
| **Main Microcontroller**: ESP32-S3 Module (Dual-core, BLE 5.0, 8MB Flash) | 1 | ESP32-S3-WROOM-1-N8 | Mouser (356-ESP32-S3-WROOM-1-N8) | $4.20 | DigiKey | $4.20 |
| **Precision Differential ADC**: 16-Bit, 4-Channel, I2C, Delta-Sigma | 1 | ADS1115IDGSR | DigiKey (296-38852-1-ND) | $4.80 | Mouser | $4.80 |
| **Analog Frontend Instrumentation Amplifier**: Low-noise, G=100, >120dB CMRR | 4 | INA128UA/2K5 | DigiKey (296-12204-5-ND) | $7.20 | Mouser | $28.80 |
| **Active Filtering Op-Amps**: Quad low-power operational amplifier (Notch/HPF) | 2 | OPA4348AIPWR | DigiKey (296-13481-1-ND) | $1.80 | Mouser | $3.60 |
| **LiPo Battery Charger IC**: Linear charger with status indicator | 1 | MCP73831T-2ATI/OT | DigiKey (MCP73831T-2ATI/OTCT-ND) | $0.85 | Mouser | $0.85 |
| **Low-Noise LDO Regulator**: 3.3V, 250mA, ultra low-noise for analog rails | 1 | LP5907MFX-3.3/NOPB | DigiKey (296-39049-1-ND) | $0.75 | Mouser | $0.75 |

### B. Discrete Hardware & Electrodes

| Component Description | Qty | Target Part Number | Primary Supplier | Unit Cost | Backup Supplier | Total Cost |
| :--- | :---: | :--- | :--- | :---: | :--- | :---: |
| **Dry Bio-compatible Electrodes**: Medical-grade stainless steel domed contacts | 10 | Stainless Steel Domed Studs | Bio-medical (SS-12MM) | $3.50 | Wearable Sensing | $35.00 |
| **Dry Contact Cable Assembly**: Flexible flat cable (FFC) with spring locks | 1 | Custom 10-ch FFC | JLCPCB | $12.00 | DigiKey | $12.00 |
| **Bone Conduction Audio Module**: 8 Ohm, 1W Transducer for local TTS feedback | 1 | Daymax Bone Conduction 1674 | Adafruit (1674) | $9.95 | Amazon | $9.95 |
| **Mono Class-D Audio Amplifier**: For driving the bone conduction module | 1 | PAM8302A Mono Amp | Adafruit (1552) | $4.95 | DigiKey | $4.95 |
| **Rechargeable Battery**: LiPo Battery (3.7V, 500mAh, JST-PH connector) | 1 | Lithium Ion Polymer 500mAh | Adafruit (1578) | $7.95 | SparkFun | $7.95 |

### C. PCB Fabrication, Enclosure, and Assembly

| Component Description | Qty | Target Part Number | Primary Supplier | Unit Cost | Backup Supplier | Total Cost |
| :--- | :---: | :--- | :--- | :---: | :--- | :---: |
| **Passives & Connectors**: SMD resistors, capacitors, USB-C jack, JST headers | 1 | passives-kit-smd | DigiKey | $15.00 | Mouser | $15.00 |
| **4-Layer PCB Manufacture & Assembly**: High-density impedance matching | 1 | Custom AFE Board | JLCPCB / PCBA | $55.00 | PCBWay | $55.00 |
| **SLS Nylon Neckband Enclosure**: Flexible TPU and PA12 structural casing | 1 | TPU/PA12 Neckband CAD | Shapeways / Protolabs | $40.00 | Hubs (3D Hubs) | $40.00 |
| **Enclosure hardware**: Brass inserts, magnetic clasps, assembly screws | 1 | casing-hardware-kit | McMaster-Carr | $9.15 | Amazon | $9.15 |
| **TOTAL FULL WEARABLE BOM** | | | | | | **$227.00** |

---

## 3. Component Sourcing and Lead Times

1. **Integrated Circuits and Passives** (DigiKey/Mouser): Ships within 24–48 hours. Zero stock risk.
2. **PCB Production & Assembly** (JLCPCB): Standard turnaround time is 6–9 calendar days (including air shipping).
3. **Neckband SLS 3D Printing** (Shapeways/Protolabs): TPU/PA12 flexible prints take approximately 5–7 business days.
