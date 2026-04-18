# Procedural Framework for a Scalable, Clean Copy-Move Forgery Dataset using MS COCO

## 1. Research Motivation and Scope
The digital image forensics community relies heavily on robust benchmark datasets to train modern Deep Neural Networks for Copy-Move Forgery Detection (CMFD). The deprecation and unavailability of the legacy USC-ISI CMFD dataset has created a significant reproducibility gap in the field.

This methodology proposes a scalable, fully local pipeline to generate exactly 50,000 high-quality forged samples utilizing the MS COCO 2017 training dataset. By relying exclusively on native annotations and standard OpenCV techniques, the pipeline prioritizes clean geometric artifact generation without introducing confounding post-processing noise.

## 2. Infrastructure and Multiprocessing Strategy
The pipeline uses `multiprocessing.Pool` to bypass Python's Global Interpreter Lock (GIL). To manage memory safely against the massive COCO annotation file, the structure loads the JSON once in the main process and parses exact polygon bounds to workers per task. 

## 3. Dataset Generation Architecture

### Core Logic
*   **Object Selection**: Extremely small objects (lack texture) or huge ones are discarded.
*   **Affine Transformations**: Continuous randomized scaling ($0.8\\times$ to $1.2\\times$) and rotation ($-45^\\circ$ to $+45^\\circ$) using `cv2.warpAffine`.
*   **Placement Strategy**: Ensured intersection over union (IoU) of 0 between source and destination patches.

## 4. Poisson Blending and Artifact Mitigation

### Ghosting Mitigation
Poisson blending depends on boundary gradients. A perfect-fit mask will read the object boundary as a 0-gradient, bleeding the destination color uncontrollably into the object boundaries ("ghosting"). To solve this, a morphological dilation (`cv2.dilate` using a $5\\times5$ kernel) fetches the original source boundary pixels, providing optimal Dirichlet conditions for a perfect blend.

### Center Offsets
The cropped object's localized centroid is explicitly computed and passed to the seamless solver, to fix coordinate mismatching.

## 5. Tri-Map Ground Truth Generation
To support dual-branch architectures (like BusterNet) for identifying both source and tampered regions simultaneously:
*   Background: 0
*   Source: 128
*   Target: 255 (Overlaid dynamically on top)
