

# Spatial Multi-omics Integration by Cross-modal Graph Contrastive Learning



## abstract



Recent advances in spatial multi-omics technologies have enabled high-resolution profiling of cellular heterogeneity while preserving spatial context, offering unprecedented opportunities to decipher tissue architecture and intercellular communication. Although existing spatial transcriptomics tools have been effective for single modal analysis, integrated interpretation of multi omics layers including spatial transcriptome, spatial proteome, and spatial epigenome remains limited due to modality specific technical biases and biological complexity. To address this, we present CoMo, a graph-based framework that synergizes multi-modal feature learning through cross attention mechanisms, coupled with dual optimization via neighbor-aware contrastive loss for cross-omics feature fusion and cluster-aware contrastive loss for spatially coherent domain identification. Evaluations on five spatial omics datasets demonstrate superior performance in spatial domain identification compared to state-of-the-art (SOTA) methods. CoMo provides a robust computational tool for multi-omics studies and supports comprehensive characterization of tissue through synergistic feature learning.



## Environment installation



**Note**: The current version of CoMo supports Linux and Windows platform.

Install packages listed on a pip file:

```
pip install -r requirements.txt
```

Install `rpy2` package:

```
pip install rpy2==3.5.10
```

Please note that the R language and the mclust package need to be installed on your system.

Install the corresponding versions of pytorch and torch_geometrics:

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 
```



## Run the code



All code is currently launched through `Tutorial.ipynb`.