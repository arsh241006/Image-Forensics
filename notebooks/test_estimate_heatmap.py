import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fusion.gradcam import estimate_heatmap

heatmap = estimate_heatmap('data/CASIA2/Tp/Tp_D_CND_S_N_txt00028_txt00006_10848.jpg')
print("Heatmap shape:", heatmap.shape)
print("Heatmap min/max:", heatmap.min(), heatmap.max())