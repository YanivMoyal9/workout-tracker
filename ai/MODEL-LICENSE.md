# AI model & runtime — licenses and attribution

## htdemucs.onnx (split here as htdemucs.onnx.part1 + htdemucs.onnx.part2)

The Demucs v4 "Hybrid Transformer" 4-stem source-separation model by Meta
(facebookresearch/demucs), converted to ONNX by the
[demucs-js](https://github.com/bakkot/demucs-js) project.

Per demucs-js's LICENSE.md: the code is MIT-licensed, but the weights file
is **not** covered by that license — it is derived from a weights file
provided by Meta, which is made available for **personal and research use
only**. This app uses it accordingly (personal, non-commercial karaoke).

## MDX ensemble model (downloaded at runtime, not stored in this repo)

The "מקסימלי" quality tier downloads UVR-MDX-NET-Inst_HQ_3.onnx directly in
the user's browser from the UVR community's public model repository
([TRvlvr/model_repo](https://github.com/TRvlvr/model_repo)) and caches it
locally in IndexedDB. Model parameters are identified via
`mdx-model-data.json`, the official parameter table from
[TRvlvr/application_data](https://github.com/TRvlvr/application_data).
The MDX inference pipeline in separate-worker.js is ported from
[nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
(MIT), which mirrors [Anjok07/ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui) (MIT).
`spark-md5.min.js` (WTFPL) computes the identification hash.

## separate-worker.js

Contains the DSP + inference pipeline adapted from
[bakkot/demucs-js](https://github.com/bakkot/demucs-js) (MIT), which ports
[facebookresearch/demucs](https://github.com/facebookresearch/demucs) (MIT).

## ort/

Unmodified runtime files from
[onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) 1.27.0 (MIT,
© Microsoft).
