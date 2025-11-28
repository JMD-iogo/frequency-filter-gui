# Frequency Filter GUI

Python/Tkinter application that allows applying frequency-domain filters  
(LPF, HPF, BPF, and BRF) to grayscale or RGB images.

---

## Features

- Load images (PNG, JPG, BMP, TIFF)
- Optional automatic conversion of RGB images to intensity
- Visualization panels:
  - Original image
  - Magnitude spectrum (FFT)
  - Phase
  - Filter mask
  - Filtered magnitude
  - Final filtered image
- Available filters:
  - **LPF** – Low-pass  
  - **HPF** – High-pass  
  - **BPF** – Band-pass  
  - **BRF** – Band-reject  
- Filter profiles:
  - Ideal  
  - Gaussian  
  - Butterworth (with adjustable order)
- **Notch filter mode**, with clickable center selection
- Real-time updates (mask and filtering)
- Automatic calculation of:
  - MSE  
  - PSNR  
- Optional saving of:
  - Filtered image  
  - Original image  
  - Original magnitude spectrum  
  - Phase  
  - Filter mask  
  - Filtered magnitude spectrum  
  - Specs text file

---

## Installation

```bash
git clone https://github.com/<your-username>/frequency-filter-gui.git
cd frequency-filter-gui

# (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
