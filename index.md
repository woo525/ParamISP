---
layout: project_page
permalink: /
title: ParamISP&#58; Learned Forward and Inverse ISPs using Camera Parameters
authors:
    Woohyeok Kim<sup>1&nbsp;*</sup> &emsp; Geonu Kim<sup>1&nbsp;*</sup> &emsp; <A href="https://junyonglee.me/">Junyong Lee</A><sup>2</sup> 
    <br><A href="https://cg.postech.ac.kr/leesy/">Seungyong Lee</A><sup>1</sup> &emsp; <A href="https://www.shbaek.com/">Seung-Hwan Baek</A><sup>1</sup> &emsp; <A href="https://www.scho.pe.kr/">Sunghyun Cho</A><sup>1</sup> 
affiliations:
    POSTECH<sup>1</sup> &emsp; Samsung AI Center Toronto<sup>2</sup>
    <br><br><p style="font-style:italic;">The IEEE/CVF Computer Vision and Pattern Recognition (CVPR) 2024</p>
paper: https://arxiv.org/abs/2312.13313
code: https://github.com/woo525/ParamISP
---

<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Abstract</h2>
        <div class="content has-text-justified">
RAW images are rarely shared mainly due to its excessive data size compared to their sRGB counterparts obtained by camera ISPs. 
Learning the forward and inverse processes of camera ISPs has been recently demonstrated, enabling physically-meaningful RAW-level image processing on input sRGB images. 
However, existing learning-based ISP methods fail to handle the large variations in the ISP processes with respect to camera parameters such as ISO and exposure time, and have limitations when used for various applications. 
In this paper, we propose ParamISP, a learning-based method for forward and inverse conversion between sRGB and RAW images, that adopts a novel neural-network module to utilize camera parameters, which is dubbed as ParamNet. 
Given the camera parameters provided in the EXIF data, ParamNet converts them into a feature vector to control the ISP networks. 
Extensive experiments demonstrate that ParamISP achieve superior RAW and sRGB reconstruction results compared to previous methods and it can be effectively used for a variety of applications such as deblurring dataset synthesis, raw deblurring, HDR reconstruction, and camera-to-camera transfer.
        </div>
    </div>
</div>

---

## Method
![overview](/static/image/overview-1.png) <span style="color:gray"> *Overview of the proposed ParamISP framework. The full pipeline is constructed by combining learnable networks (ParamNet, LocalNet, GlobalNet) with invertible canonical camera operations (CanoNet). CanoNet consists of differentiable operations without learnable weights, where WB and CST denote white balance and color space transform, respectively.* </span>
<br/><br/>

![paramnet](/static/image/paramnet-1.png) <span style="color:gray"> *Architecture of ParamNet. (a) Given camera optical parameters, ParamNet estimates optical parameter features used for modulating the LocalNet and GlobalNet. (b) In order to deal with different scales and non-linearly distributed values of optical parameters, we propose to use non-linear equalization that exploits multiple non-linear mapping functions.* </span>
<br/><br/>

> Given a target camera, our goal is to learn its forward and inverse ISP processes that change with respect to camera parameters. To accomplish this, ParamISP is designed to have a pair of forward (RAW-to-sRGB) and inverse (sRGB-to-RAW) ISP networks. Both networks are equipped with ParamNet so that they adaptively operate based on camera parameters. In ParamISP, we classify camera parameters into two distinct categories: optical parameters (including exposure time, sensitivity, aperture size, and focal length) and canonical parameters (Bayer pattern, white balance coefficients, and a color correction matrix). To harness the canonical parameters, our ISP networks incorporate CanoNet, a subnetwork that performs canonical ISP operations without learnable weights. For the optical parameters, we introduce ParamNet, which is the key component to dynamically control the behavior of the ISP networks based on the optical parameters.

## Results
#### Qualitative
###### *1. Inverse (sRGB <span style="font-size:200%">&rarr;</span> RAW)*
![inverse](/static/image/inverse-1.png)

###### *2. Forward (RAW <span style="font-size:200%">&rarr;</span> sRGB)*
![forward](/static/image/forward-1.png)

#### Quantitative
![fwdinvQuan](/static/image/fwdinvQuan-1.png)

## Applications
#### *1. RAW Deblurring* 
![rawdeblur](/static/image/rawdeblur-1.png)

#### *2. Deblurring Dataset Synthesis*
![deblurdataset](/static/image/deblurdataset-1.png)

#### *3. HDR Reconstruction*
               ![hdr](/static/image/hdr-1.png){: width="50%"}

#### *4. Camera-to-Camera Transfer*
               ![cam2cam](/static/image/cam2cam-1.png)

## Citation
```
@inproceedings{kim2024paramisp,
  title     = {ParamISP: Learned Forward and Inverse ISPs using Camera Parameters},
  author    = {Woohyeok Kim, Geonu Kim, Junyong Lee, Seungyong Lee, Seung-Hwan Baek, Sunghyun Cho},
  booktitle = {The IEEE/CVF Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}
```
