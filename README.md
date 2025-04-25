# DTMPI-DIVR: Digital Twins for Multi-margin Physical Interaction

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![MATLAB](https://img.shields.io/badge/MATLAB-R2021a%2B-orange)

Repository for the paper **"DTMPI-DIVR: A Digital Twins for Multi-margin Physical Information via Dynamic Interaction of Virtual and Real Sound-vibration Signals for Bearing Fault Diagnosis without Real Fault Samples"**.

🔔 **Notice:** This repository contains partial implementation code and sample datasets. The full code will be released upon paper acceptance. For research collaborations or early access, please contact the corresponding author.

---

## 📖 Table of Contents
- [Introduction](#-introduction)
- [Features](#-features)
- [Dataset Structure](#-dataset-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Code Structure](#-code-structure)
- [Contribution](#-contribution)
- [Citation](#-citation)
- [License](#-license)

---

## 🌟 Introduction
This repository implements a digital twin framework for bearing fault diagnosis using dynamic interaction between virtual and real-world sensor signals. Key innovations include:
- 15-DOF vibration dynamics modeling
- Cross-domain physical information coupling
- Fault diagnosis without real fault samples
- Multi-modal data fusion (audio/vibration)

---

## 🚀 Features
- **Digital Twin Core**
  - `15-DOF vibration modeling` (`十五自由度振动响应动力学建模.py`)
  - Physical coupling algorithms (`some_coupling_coefficient.m`)
  
- **Dynamic Interaction**
  - DL-based proxy models (`基于DL代理的虚拟与现实动态交互方法.py`)
  - ELM-based interaction (`基于ELM的虚拟与现实动态交互方法.py`)
  - FA optimization (`基于FA的虚拟与现实动态交互方法.py`)

- **Diagnosis Toolkit**
  - Data preprocessing (DigitalBearingDataBase.py)
  - Twin-function library (DigitalTwinsFunctions.py)
  - Fault diagnosis module (`数字孪生故障诊断.py`)

---

## 📂 Dataset Structure
### File Naming Convention
