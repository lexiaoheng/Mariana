

# Mariana: Open-Sourse Seismic Toolbox 

- Mariana V0.1.0: Seismic toolbox with some published works on seismic reconstruction, noise attenuation and inversion.  
- This repository is supported and maintained for a long time, so please issue if you run into problems.
- Welcome to participate in the construction.

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GPL-3.0 License][license-shield]][license-url]

---
<br />
<p align="center">
  <h3 align="center"><span style="color: #2D3C81;"> Mariana：Open-Source Seismic Toolbox</span></h3>
  <p align="center">
    Seismic Noise Attenuation, Reconstruction and Inversion 
    <br />
    <a href="https://github.com/lexiaoheng/Mariana/tree/main/NoiseAttenuation/"><strong>Explore seismic processing algorithms></strong></a>
    <br />
    <br />
    <a href="https://github.com/lexiaoheng/Mariana/tree/main/main_workflow.m">Start</a>
    ·
    <a href="https://github.com/lexiaoheng/Mariana/issues">Report bug</a>
    ·
    <a href="https://github.com/lexiaoheng/Mariana/issues">Issues</a>
  </p>

<p align="center"><h2 align="center"><span style="color: #FF5733;"> Catalogue </span></h2>

### <span style="color: #2D3C81;"> 1. Setup </span>
  - [Environment](#environment)
  - [Quick Start](#quick-start)

### <span style="color: #2D3C81;"> 2. Seismic Noise Attenuation </span>
  - <b>Supervised Deep Learning</b> 
    - [Diffusion Model](#diffusion-model)
  - <b>Unsupervised/Zero-shot Deep Learning</b>  
    - [Adaptive Convolutional Filter](#adaptive-convolutional-filter)
    - [Convolutional Autoencoder](#convolutional-autoencoder)
    
### <span style="color: #2D3C81;"> 3. Seismic Reconstruction </span>
  - <b>Unsupervised/Zero-shot Deep Learning</b>
    - [Zero-shot Self-Consistency Learning](#zero-shot-self-consistency-learning)

### <span style="color: #2D3C81;"> 4. Inversion </span>
  - <b>Post-Stacked Inversion</b>
    - [Encoder-Inverter Framework](#encoder-inverter-framework)
    - [Dual-Banch Double Inversion](#dual-branch-double-inversion)

### <span style="color: #2D3C81;"> 5. Utils </span>
  - [Evaluation](#evaluation)
  - [Noise Level Estimate](#noise-level-estimate)

<p align="center"><h2 align="center"><span style="color: #FF5733;"> Mariana Docs</span></h2>

<table><td>

## <span style="color: #2D3C81;"> Setup </span>

### **Environment**

1. Anaconda (Make sure to configure it into an environment variable)
2. Anaconda environment (Python 3.7.11 is recommended)
3. CUDA and cuDNN (Select the appropriate version according to the requirements.txt of project you need)

### **Quick Start**

```python
from Mariana_Core import NoiseAttenuation, Inversion, Reconstruction, utils

# the list of APIs

## Noise Attenuation
NoiseAttenuation.CAE(data, hidden_dim=4)
NoiseAttenuation.ACF(data)
NoiseAttenuation.DPM(data, ddim=True)

## Reconstruction
Reconstruction.SCL(data, missing_mask)

## Inversion
Inversion.EIF(data, well_log)
Inversion.DBDI(data, well_log)

## utils
utils.evaluate(label, prediction, method)
utils.noise_level_estimate(data)
```

</td></table>

## <b><span style="color: #2D3C81;"> Seismic Noise Attenuation </span></b>

<table><td>

- ### **Diffusion model**
  - #### Classification:
  
    - Seismic Noise Attenuation (Pre-Stacked and Post-Stacked Data, 2-D only)
    - Supervised Deep Learning Methods (With Pre-trained Model)

  - #### Abstract:

    - Diffusion model for seismic strong noise attenuation with principal component analysis for noise level (t in diffusion model) estimation.
    - There are two parts: 1. diffusion model; 2. fast diffusion model. 

  - #### Usage:

    ```python
    from Mariana_Core.NoiseAttenuation import DPM
    output = DPM(data, ddim=True)
    ```
    - <b><span style="color: #2D3C81;"> data (Numpy.ndarray) </span></b> - Noisy seismic data. If processing a single data, its shape should be <b>[128, 128]</b>. And the function also supports processing multiple data, with the input data shape being <b>[128, 128, data_num]</b>.
    - <b><span style="color: #2D3C81;"> ddim (bool, optional) </span></b> - If <b>True</b>, using fast diffusion model. Default: <b>True</b>.
    - <b><span style="color: #2D3C81;"> output (Numpy.ndarray) </span></b> - Output seismic data, has the same shape as the input data.
  - #### Reference:

    - [Peng J, Li Y, Liu Y, et al. Fast diffusion model for seismic data noise attenuation. Geophysics, 2025, 90(4): 1-55.](https://library.seg.org/doi/abs/10.1190/geo2024-0187.1)  
    - [Peng J, Li Y, Liao Z, et al. Seismic data strong noise attenuation based on diffusion model and principal component analysis. IEEE Transactions on Geoscience and Remote Sensing, 2024, 62: 1-11.](https://ieeexplore.ieee.org/abstract/document/10409237)

</td></table>

<table><td>

- ### **Adaptive Convolutional Filter**

  - #### Classification:
  
    - Seismic Noise Attenuation (Pre-Stacked and Post-Stacked Data)
    - Zero-shot Deep Learning Methods

  - #### Abstract:

    - Adaptive convolutional filter is a deep learning methods driving by multiple priors instead of extenal datasets.
    - Constructed by three convolutional layers, which have total 2464 learnable parameters. 
  
  - #### Usage:

    ```python
    from Mariana_Core.NoiseAttenuation import ACF
    output = ACF(data)
    ```
    - <b><span style="color: #2D3C81;"> data (Numpy.ndarray) </span></b> - Noisy seismic data, 2D only.
    - <b><span style="color: #2D3C81;"> output (Numpy.ndarray) </span></b> - Output seismic data, has the same shape as the input data.

  - #### Reference:

    - [Peng J, Li Y, LIu Y, et al. Adaptive Convolutional Filter for Seismic Noise Attenuation. arXiv preprint arXiv:2410.18896, 2024.](https://arxiv.org/pdf/2410.18896)

</td></table>

<table><td>

- ### **Convolutional Autoencoder**

  - #### Classification:
  
    - Seismic Noise Attenuation (Pre-Stacked and Post-Stacked Data)
    - Unsupervised Deep Learning Methods

  - #### Abstract:

    - Convolutional autoencoder is a classic unsupervised deep learning framework widely used in the fields of noise processing and feature extraction.
    - The model used in this project consists of 8 convolutional layers, with each layer using a convolutional kernel size of <b>[4, 4]</b> and a sliding step size of <b>[2, 2]</b>
  
  - #### Usage:

    ```python
    from Mariana_Core.NoiseAttenuation import CAE
    output = CAE(data, hidden_dim)
    ```
    - <b><span style="color: #2D3C81;"> data (Numpy.ndarray) </span></b> - Noisy seismic data, 2D only. Due to the use of a fixed convolution kernel size and sliding step, the length and width of the input data must be an integer multiple of <b>16</b>.
    - <b><span style="color: #2D3C81;"> hidden_dim (int) </span></b> - Hidden dimension of feature map. 
    - <b><span style="color: #2D3C81;"> output (Numpy.ndarray) </span></b> - Output seismic data, has the same shape as the input data.

</td></table>

## <b><span style="color: #2D3C81;"> Reconstruction </span></b>

<table><td>

- ### **Zero-Shot Self-Consistency Learning**
  - #### Classification:
  
    - Seismic Reconstruction (Post-Stacked Data, 2-D only)
    - Zero-Shot Deep Learning Methods

  - #### Abstract:

    - Zero-shot self-consistency learning assumes predictability between reconstructed data and sampled data, without the need for extenal datasets for supervised training. 
    - Using a lightweight convolutional autoencoder.
  
  - #### Usage:

    ```python
    from Mariana_Core.Reconstruction import SCL
    output = SCl(data, miss_mask, mode='zs-scl')
    ```
  
    - <b><span style="color: #2D3C81;"> data (Numpy.ndarray) </span></b> - Sampled seismic data, 2D only. 
    - <b><span style="color: #2D3C81;"> missing_mask (Numpy.ndarray) </span></b> - Matrix for marking missing data. If the seismic trace is missing, the corresponding positions in the mssing_mask are all <b>0</b>.
    - <b><span style="color: #2D3C81;"> mode (str, optional) </span></b> - if <b>'zs-scl'</b>, reconstruct seismic data with zero-shot self-consistency learning. if <b>'traditional'</b>, using the mean square error loss function and constrain only the sampled data (similar to noise attenuation).
    - <b><span style="color: #2D3C81;"> output (Numpy.ndarray) </span></b> - Output seismic data, has the same shape as the input data.

  - #### Reference:

    - [Peng J, Liu Y, Wang M, et al. Zero-Shot Self-Consistency Learning for Seismic Irregular Spatial Sampling Reconstruction. arXiv preprint arXiv:2411.00911, 2024.](https://arxiv.org/pdf/2411.00911)  
 
</td></table>


## <b><span style="color: #2D3C81;"> Inversion </span></b>

<table><td>

- ### **Encoder-Inverter Framework**
  - #### Classification:
  
    - Seismic Acoustic Impedance Inversion (Post-Stacked Data, 2-D only)
    - Few-Shot Deep Learning Methods

  - #### Abstract:

    - Firstly, the seismic data is encoded into high dimension linear features through an encoder, and the inverter is fine-tuned through few well-logging data. The use of high dimensional linear features can enhance the extrapolation ability of inverters when processing non-well data.
  
  - #### Usage:

    ```python
    from Mariana_Core.Inversion import EIF
    output = EIF(data, well_log)
    ```
  
    - <b><span style="color: #2D3C81;"> data (Numpy.ndarray) </span></b> - Seismic data, 2D only. 
    - <b><span style="color: #2D3C81;"> well_log (Numpy.ndarray) </span></b> - Sparse well-logging matrix, has the same shape as the seismic data. If without well-logging data, the corresponding positions in the mssing_mask are all <b>0</b>.
    - <b><span style="color: #2D3C81;"> output (Numpy.ndarray) </span></b> - Inverted acoustic impedance, has the same shape as the seismic data.

  - #### Reference:

     - [Peng J, Liu Y, Wang M, et al. Encoder-Inverter Framework for Seismic Acoustic Impedance Inversion. arXiv preprint arXiv:2507.19933, 2025.](https://arxiv.org/abs/2507.19933)  

</td></table>

<table><td>

- ### **Dual-Branch Double Inversion**
  - #### Classification:
  
    - Seismic Acoustic Impedance Inversion (Post-Stacked Data, 2-D only)
    - Few-Shot Deep Learning Methods

  - #### Abstract:

    - Dual networks are used for joint training. The inversion network is used for the inversion of the acoustic impedance from seismic data, and the forward network learns the mapping from acoustic impedance to seismic data, so that there is no need to extract the seismic wavelets.
    - The model adopts a two-branch structure to effectively extract the characteristics of long seismic time series.    

  - #### Usage:

    ```python
    from Mariana_Core.Inversion import DBDI
    output = DBDI(data, well_log)
    ```
  
    - <b><span style="color: #2D3C81;"> data (Numpy.ndarray) </span></b> - Seismic data, 2D only. 
    - <b><span style="color: #2D3C81;"> well_log (Numpy.ndarray) </span></b> - Sparse well-logging matrix, has the same shape as the seismic data. If without well-logging data, the corresponding positions in the mssing_mask are all <b>0</b>.
    - <b><span style="color: #2D3C81;"> output (Numpy.ndarray) </span></b> - Inverted acoustic impedance, has the same shape as the seismic data.

  - #### Reference:

    - [Feng W, Liu Y, Li Y, et al. Acoustic impedance prediction using an attention-based dual-branch double-inversion network. Earth Science Informatics, 2025, 18(1): 70.](https://link.springer.com/article/10.1007/s12145-024-01548-4)  

</td></table>

## <b><span style="color: #2D3C81;"> Utils </span></b>

<table><td>

- ### **Evaluation**

  - #### Abstract:

    - Function for computing various metrics, can evaluate 2D, 3D or higher dimension data.
  
  - #### Usage:

    ```python
    from Mariana_Core.utils import evaluate
    metric = evaluate(label, prediction, methods='MAE')
    ```
  
    - <b><span style="color: #2D3C81;"> label (Numpy.ndarray) </span></b> - Ground truth. 
    - <b><span style="color: #2D3C81;"> prediction (Numpy.ndarray) </span></b> - Processed data, has the same shape as the ground truth.
    - <b><span style="color: #2D3C81;"> method (str, optinal) </span></b> - Calculated indicators. 
      - <b>'MAE'</b> - Mean absolute error
      - <b>'MSE'</b> - Mean square error
      - <b>'NMSE'</b> - Normalized mean square error
      - <b>'SNR'</b> - Signal-to-noise ratio
      - <b>'PCC'</b> - Pearson correlation coefficient
      - <b>'SSIM'</b> - Structural similarity (normalized)
      - <b>'R2'</b> - R-square fitting coefficient

</td></table>

<table><td>

- ### **Noise Level Estimate**

  - #### Abstract:

    - Used to evaluate the noise level (variance of noise) of data without ground truth. 
  
  - #### Usage:

    ```python
    from Mariana_Core.utils import noise_level_estimate
    variance = noise_level_estimate(data, output)
    ```
    - <b><span style="color: #2D3C81;"> data (Torch.tensor) </span></b> - Noisy dataset in <b>Torch.tensor</b>, with a shape of <b>[B, C, H, W]</b>.
    - <b><span style="color: #2D3C81;"> output (str, optional) </span></b> - If <b>'mean'</b>, return mean variance on batch, else return variance with a shape of <b>[B, ]</b>.
    - <b><span style="color: #2D3C81;"> variance (Torch.tensor) </span></b> - Estimated noise variance has a shape of <b>[1]</b> or <b>[B, ]</b>.

</td></table>






<!-- links -->
[your-project-path]: lexiaoheng/Mariana
[contributors-shield]: https://img.shields.io/github/contributors/lexiaoheng/Mariana.svg?style=flat-square
[contributors-url]: https://github.com/lexiaoheng/Mariana/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/lexiaoheng/Mariana.svg?style=flat-square
[forks-url]: https://github.com/lexiaoheng/Mariana/network/members
[stars-shield]: https://img.shields.io/github/stars/lexiaoheng/Mariana.svg?style=flat-square
[stars-url]: https://github.com/lexiaoheng/Mariana/stargazers
[issues-shield]: https://img.shields.io/github/issues/lexiaoheng/Mariana.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/lexiaoheng/Mariana.svg
[license-shield]: https://img.shields.io/github/license/lexiaoheng/Mariana.svg?style=flat-square
[license-url]: https://github.com/shaojintian/Best_README_template/blob/master/LICENSE.txt





