# Combi-CAM: A Novel Multi-Layer Approach for Explainable Image Geolocalization

Official repository for Combi-CAM: A Novel Multi-Layer Approach for Explainable Image Geolocalization

Combi-CAM generalizes Grad-CAM by aggregating gradient-weighted activation maps across all stages of the CNN instead of relying only on the final layer. This yields sharper object localization, improved interpretability, and deeper insight into how geolocalization networks combine low-, mid-, and high-level features.

![Explainability Methods](https://github.com/DavidFaget/Combi-CAM/blob/main/images/cams.png)

## 🚀 Inference & Explainability

Run inference with any combination of explainability methods:

```bash
!python inference.py --image_url "YOUR_IMAGE_URL" --use_cpu --layercam --gradcam --gradcamplusplus --scorecam --combicam
```

Recommended to generate figures one by one to avoid RAM overload.

You may also use:
```bash
--image_path "local_image.jpg"
```
⚠️ Recommendation:
Generate CAM visualizations one at a time to avoid RAM overload.

## 📂 Repository Structure

```
Combi-CAM/
│
├── data/ # CERTH geolocalization model assets (proprietary)
│ ├── cells_assignments.pkl
│ └── initial_mu_kappa.pkl
│
├── images/ # Images used in the README
│
├── notebooks/
│ └── Combi-CAM_playground.ipynb # Playground notebook for inference & visualization
│
├── src/ # Main source code
│ ├── modules.py # Model components + Combi-CAM implementation
│ └── utils.py # Utility functions
│
├── README.md
├── LICENSE.txt # Defines the License
├── inference.py # Inference script (supports all CAM methods)
└── requirements.txt # Python dependencies
```

## 📜 Citation

If you use Combi-CAM in your research, please cite the associated paper Combi-CAM: A Novel Multi-Layer Approach for Explainable Image Geolocalization (VISAPP2026)
VISAPP 2026

## 🛡 Copyright
This repository uses the geolocalization model developed by the CERTH team. All rights and credits for the geolocalization model belong to CERTH. Combi-CAM is an original contribution introduced in the associated paper.

## 📦 Model Weights
This repository relies on the geolocalization model developed by CERTH based on EfficientNet-B4.
The weights are proprietary and cannot be distributed.

To request access, please contact the authors of:

[1] Giorgos Kordopatis-Zilos, Panagiotis Galopoulos, Symeon Papadopoulos, and Ioannis Kompatsiaris. 2021. Leveraging EfficientNet and Contrastive Learning for Accurate Global-scale Location Estimation. In Proceedings of the 2021 International Conference on Multimedia Retrieval (ICMR '21). Association for Computing Machinery, New York, NY, USA, 155–163. https://doi.org/10.1145/3460426.3463644
