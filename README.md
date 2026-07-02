# PDE Foundation Model Exploration

ARIA Internship project (5 January 2026 - 3 July 2026) supervised by Dr Ooi Chin Chun and Dr Wei Zhao.

Explored various foundation models (DPOT, MORPH, PDEformer-2, PDEtransformer, PROSE-FD) and documented the various results we had with inference on various datasets (PDEBench, PDEArena, CFDBench, etc).

For all models, we attempted to convert new/unseen datasets into the format of various pretrained datasets used by the respective models. Results of inferences on the converted datasets are recorded and located in the folders of each model.

For DPOT specifically, a pseudo-inverse function written by Dr Wei Zhao was added to the last layer of the model. Inference was done using this variant of DPOT on the SGEDULION/SGMERLION/SGLION datasets. Results are recorded as well.