# Combi-CAM: Explainable Geolocalization for Autonomous Driving

Official repository for Combi-CAM: Explainable Geolocalization for Autonomous Driving paper.


![Heading of the paper](https://github.com/DavidFaget/Combi-CAM/blob/main/examples/images/paper.png)

## Inference with explainability methods:

```bash
!python inference.py --image_url "..." --use_cpu --layercam --gradcam --gradcamplusplus --scorecam --combicam
```

Recommended to generate figures one by one to avoid RAM overload.

It is also possible to use --image_path instead of --image_url.

## Additional Talk2Car Examples
Additional illustrations of the application of Combi-CAM on Talk2Car [2] dataset images can be found in the examples folder.

## Copyright
This repository uses the geolocation model developed by the CERTH team. All rights and credits for the geolocation model belong to CERTH.

## Model Weights
The model weights are proprietary to CERTH and will not be released. For access to the weights, please reach out to the authors of [1].

## References
[1] Giorgos Kordopatis-Zilos, Panagiotis Galopoulos, Symeon Papadopoulos, and Ioannis Kompatsiaris. 2021. Leveraging EfficientNet and Contrastive Learning for Accurate Global-scale Location Estimation. In Proceedings of the 2021 International Conference on Multimedia Retrieval (ICMR '21). Association for Computing Machinery, New York, NY, USA, 155–163. https://doi.org/10.1145/3460426.3463644

[2] Deruyttere, Thierry & Vandenhende, Simon & Grujicic, Dusan & Van Gool, Luc & Moens, Marie-Francine. (2019). Talk2Car: Taking Control of Your Self-Driving Car. 2088-2098. 10.18653/v1/D19-1215. 
