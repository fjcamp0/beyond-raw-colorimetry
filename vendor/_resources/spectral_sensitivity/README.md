# Smartphone Spectral Sensitivity Database

**Source:** Ohlab Mobile Spectral Sensitivity Database  
**URL:** https://ohlab.kic.ac.jp/index/mobilespec_database

## Citation (REQUIRED when using these files)

> Tominaga, S.; Nishi, S.; Ohtera, R.  
> **Measurement and Estimation of Spectral Sensitivity Functions for Mobile Phone Cameras**  
> *Sensors* **2021**, 21(15), 4985.  
> https://doi.org/10.3390/s21154985

Cite this paper in any publication, presentation, or report that uses these
spectral sensitivity data files.

---

## File format

Each `.txt` file contains **31 rows × 3 tab-separated columns**.

- **Rows:** wavelengths 400, 410, 420, … 700 nm (10 nm steps, 31 points)
- **Columns:** R sensitivity, G sensitivity, B sensitivity (normalised, peak ≈ 1)
- No header row.

```
wavelength_nm = range(400, 710, 10)   # 400, 410, ..., 700
```

Example (iphone_6s.txt, first 3 rows = 400, 410, 420 nm):
```
2.42E-02    2.65E-02    8.75E-02    ← 400 nm: low R, low G, moderate B
3.18E-02    2.84E-02    2.68E-01    ← 410 nm
3.81E-02    3.21E-02    4.66E-01    ← 420 nm
```

---

## Available devices

| File | Device |
|------|--------|
| `iphone_6s.txt` | Apple iPhone 6s |
| `iphone_se.txt` | Apple iPhone SE |
| `iphone_8.txt` | Apple iPhone 8 |
| `iphone_x.txt` | Apple iPhone X |
| `iphone_11.txt` | Apple iPhone 11 |
| `iphone_12_pro_max.txt` | Apple iPhone 12 Pro MAX (normal lens) |
| `huawei_p10_lite.txt` | Huawei P10 lite |
| `huawei_nova_lite2.txt` | Huawei nova lite 2 |
| `samsung_galaxy_s7_edge.txt` | Samsung Galaxy S7 edge |
| `samsung_galaxy_s9.txt` | Samsung Galaxy S9 |
| `samsung_galaxy_note10plus.txt` | Samsung Galaxy Note10+ |
| `samsung_galaxy_s20.txt` | Samsung Galaxy S20 |
| `sharp_aquos_sense3_lite.txt` | Sharp AQUOS sense3 lite |
| `sharp_aquos_r5g.txt` | Sharp AQUOS R5G |
| `xiaomi_mi_mix_2s.txt` | Xiaomi Mi Mix 2s |
| `xiaomi_redmi_note_9s.txt` | Xiaomi Redmi Note 9S |
| `sony_xperia_1_ii.txt` | Sony Xperia 1 II |
| `sony_xperia_5_ii.txt` | Sony Xperia 5 II |
| `fujitsu_arrows_nx9.txt` | Fujitsu arrows NX9 |
| `google_pixel_4.txt` | Google Pixel 4 |
| `average_all.txt` | Average of all devices (use when camera unknown) |

---

## Usage in raw-image-analysis

Load via `_resources/spectral_utils.py`:

```python
from raw_image_analysis._resources.spectral_utils import load_sensitivity, spectral_to_rgb_weights

# Load for a specific device
sensitivity = load_sensitivity("iphone_11")
# sensitivity.shape = (31, 3) — rows: 400–700 nm, cols: R, G, B

# Use average when camera model is unknown
sensitivity = load_sensitivity("average_all")
```

When camera model is unknown, use `average_all.txt`.
