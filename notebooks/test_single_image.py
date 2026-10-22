import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fusion.pipeline import ForensicsPipeline

pipeline = ForensicsPipeline()

# replace with the real full path from the Get-ChildItem output above
img_path = 'data/CASIA2/Au/Au_ani_00001.jpg'

label, prob = pipeline.predict(img_path)
print(f"Predicted class: {label} (0=authentic, 1=tampered)")
print(f"Tampered probability: {prob:.4f}")