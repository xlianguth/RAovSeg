# Image Analysis Tools using for RAovSeg

## Introduction

This repository provides a set of useful image analysis tools for image segmentation tasks, particularly developed for the RAovSeg pipeline.

---

## Requirements

- Python 3.9  
Please ensure the following Python libraries are installed:

- `numpy`, `torch`  
- `SimpleITK`, `scipy`

You can install them via:

```bash
pip install numpy torch SimpleITK scipy
```

### Dataset
- We provide a sample image and segmentation mask as a tutorial example.
- Dataset source: UTHealth - Endometriosis MRI Dataset (UT-EndoMRI). (https://zenodo.org/records/13749613)

---

## Tutorial

Run tutorial.py to test the tools using the provided example image.

```bash
python tutorial.py
```

This will generate sample outputs such as:
![img](./example/results/Image.png)

---

**Note:** 

The RAovSeg pipeline is implemented using MONAI's deep learning models:

- ResNet: https://docs.monai.io/en/stable/networks.html#resnet
- Attention Unet: https://docs.monai.io/en/stable/networks.html#attentionunet

The Loss Functions Used:

- BCEWithLogitsLoss: https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
- Focal Tversky Loss: https://github.com/nabsabraham/focal-tversky-unet


## Reference
- Liang, X., Alpuing Radilla, A. L., Khalaj, K., & Giancardo, L., et al. (2025) A Multi-Modal Pelvic MRI Dataset for Deep Learning-Based Pelvic Organ Segmentation in Endometriosis. Scientific Data. (Under review).
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
- Oktay, O., Schlemper, J., Folgoc, L. L., Lee, M., Heinrich, M., Misawa, K., ... & Rueckert, D. (2018). Attention u-net: Learning where to look for the pancreas. arXiv preprint arXiv:1804.03999.
- Abraham, N., & Khan, N. M. (2019, April). A novel focal tversky loss function with improved attention u-net for lesion segmentation. In 2019 IEEE 16th international symposium on biomedical imaging (ISBI 2019) (pp. 683-687). IEEE.

## Acknowledgment
The creators of this dataset are:
Liang, X., Alpuing Radilla, L. A., Khalaj, K., Mokashi, C., Guan, X., Roberts, K. E., Sheth, S. A., Tammisetti, V. S., & Giancardo, L. (2024). UTHealth - Endometriosis MRI Dataset (UT-EndoMRI) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13749613

