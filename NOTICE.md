# Attribution

The detector, Xception backbone, trainer, losses, metrics and dataset pipeline
are derived from DeepfakeBench, originally authored by Zhiyuan Yan and
contributors. The upstream license is retained verbatim in `LICENSE`.

- DeepfakeBench: https://github.com/SCLBD/DeepfakeBench
- `scripts/dlib_alignment.py` retains the DeepfakeBench
  `get_keypts` and `extract_aligned_face_dlib` functions unchanged. The training
  preparation entry point adds fixed-index selection, validation, reporting,
  and resumable output writing.
- Xception / FaceForensics++ detector: Rossler et al., ICCV 2019.
- FFD: Dang et al., On the Detection of Digital Face Manipulation, CVPR 2020.
  The ten MCT template images are retained for the FFD implementation.
- SPSL: Liu et al., Spatial-Phase Shallow Learning, CVPR 2021.

CoC augmentation, benchmark configurations and public data preparation accompany
the Caught on Camera paper. Public entry points and data-layout adaptations retain
the paper's active preprocessing and detector behavior. Model weights and source
datasets carry their respective terms; the code license does not replace them.
