from pathlib import Path

import rawpy

DATA_ROOT = Path(__file__).resolve().parent.parent / "Data"
fe = DATA_ROOT / "images/ferricyanide/led_set2/Blank_IMG_260507_122659.dng"
with rawpy.imread(str(fe)) as raw:
    print("raw sizes:", raw.sizes)
    rgb = raw.postprocess(use_camera_wb=True, output_bps=8, no_auto_bright=True)
print("postprocess shape (H, W, C):", rgb.shape)
