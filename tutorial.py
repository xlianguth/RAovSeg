#%%
import SimpleITK as sitk
import RAovSeg_tools as tools
import numpy as np
import matplotlib.pyplot as plt

# Load the image example
Img = sitk.ReadImage('./example/UTEndoMRI_example.nii.gz',sitk.sitkFloat64)
# Image resampling
Img = tools.ImgResample(Img, out_spacing=(0.35, 0.35, 6.0), out_size=(512,512,38), is_label=False, pad_value=0)
Img = sitk.GetArrayFromImage(Img)
# Image normalization
Img = tools.ImgNorm(Img,norm_type="percentile_clip",percentile_low=1,percentile_high=99)
# Image preprocessing
Img_preprossed = tools.preprocess_(Img, o1=0.24, o2=0.3)

# Plot and compare the original and prorocessed image
plt.subplot(1,2,1)
plt.imshow(Img[18], cmap='gray')
plt.title("Original")
plt.subplot(1,2,2)
plt.imshow(Img_preprossed[18], cmap='gray')
plt.title("Preprocessed")
plt.show()

# Load the Segmentation an Model Prediction
Lb = sitk.ReadImage('./example/OvLabel.nii.gz',sitk.sitkInt32)
Lb = sitk.GetArrayFromImage(Lb)
Pred = sitk.ReadImage('./example/Prediction.nii.gz',sitk.sitkInt32)
Pred = sitk.GetArrayFromImage(Pred)
Pred_postprocessed = tools.postprocess_(Pred)

plt.figure(figsize=(9, 3)) 
plt.subplot(1, 3, 1)
plt.imshow(Lb[18], cmap='gray')  
plt.title("Ground Truth")
plt.subplot(1, 3, 2)
plt.imshow(Pred[18], cmap='gray')
plt.title("Prediction")
plt.subplot(1, 3, 3)
plt.imshow(Pred_postprocessed[18], cmap='gray') 
plt.title("Postprocessed")
plt.show()

# Dice calculation
dsc1 = tools.dsc_cal_np(Pred,Lb) 
dsc2 = tools.dsc_cal_np(Pred_postprocessed,Lb)
print(f"The DSC between groundtruth and prediction is {dsc1}")
print(f"The DSC between groundtruth and postprocessed prediction is {dsc2}")

#%%