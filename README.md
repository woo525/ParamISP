## ParamISP: Learned Forward and Inverse ISPs using Camera Parameters<br><sub>Official PyTorch Implementation of the CVPR 2024 Paper</sub>

*Woohyeok Kim\*, Geonu Kim\*, Junyong Lee, Seungyong Lee, Seung-Hwan Baek, Sunghyun Cho<br>*

[\[Paper\]](https://arxiv.org/abs/2312.13313)
[\[Project Page\]](https://woo525.github.io/ParamISP/)

![overview](https://github.com/woo525/ParamISP/assets/32587029/b4bb291f-14e4-42dd-8642-518752843cc3)

### Abstract

RAW images are rarely shared mainly due to its excessive data size compared to their sRGB counterparts obtained by camera ISPs. Learning the forward and inverse processes of camera ISPs has been recently demonstrated, enabling physically-meaningful RAW-level image processing on input sRGB images. However, existing learning-based ISP methods fail to handle the large variations in the ISP processes with respect to camera parameters such as ISO and exposure time, and have limitations when used for various applications. In this paper, we propose ParamISP, a learning-based method for forward and inverse conversion between sRGB and RAW images, that adopts a novel neural-network module to utilize camera parameters, which is dubbed as ParamNet. Given the camera parameters provided in the EXIF data, ParamNet converts them into a feature vector to control the ISP networks. Extensive experiments demonstrate that ParamISP achieve superior RAW and sRGB reconstruction results compared to previous methods and it can be effectively used for a variety of applications such as deblurring dataset synthesis, raw deblurring, HDR reconstruction, and camera-to-camera transfer.

### Environment Setting
* Ubuntu 20.04
* Python 3.10
* PyTorch 1.12.1

      pip install -r requirements.txt

### Training
As described in the paper, ParamISP is trained in two stages for both the inverse and forward directions: pre-training and fine-tuning. Additionally, before applying it to applications, further joint fine-tuning can be conducted. We provide a small dataset example and the official weights reported in the paper to enable the execution of the code. You can set the dataset path through the **.env** file.

[\[Dataset example\]](https://drive.google.com/drive/folders/1ZCi3ZXLeM7Ary6eWlVaVTTDm-kWcXWjU?usp=sharing) [\[Official weights\]](https://drive.google.com/drive/folders/1ZCi3ZXLeM7Ary6eWlVaVTTDm-kWcXWjU?usp=sharing)
#### 1. Pre-training

        CUDA_VISIBLE_DEVICES=0 python models/paramisp.py -o demo --inverse train

#### 2. Fine-tuning
        
        CUDA_VISIBLE_DEVICES=0 python models/paramisp.py -o demo --inverse train --camera D7000

#### 3. Joint fine-tuning

        CUDA_VISIBLE_DEVICES=0,1 python models/paramisp_joint.py -o demo train --camera D7000 --pisp-inv weights/fine_tuning/inverse/D7000.ckpt --pisp-fwd weights/fine_tuning/forward/D7000.ckpt



### Test
 
        CUDA_VISIBLE_DEVICES=0 python models/paramisp.py -o demo --inverse test --ckpt weights/fine_tuning/inverse/D7000.ckpt --camera D7000

### Inference
 
        CUDA_VISIBLE_DEVICES=0 python models/paramisp.py -o demo --inverse predict --ckpt weights/fine_tuning/inverse/D7000.ckpt --camera D7000

### Citation
```
@inproceedings{kim2024paramisp,
  title     = {ParamISP: Learned Forward and Inverse ISPs using Camera Parameters},
  author    = {Woohyeok Kim, Geonu Kim, Junyong Lee, Seungyong Lee, Seung-Hwan Baek, Sunghyun Cho},
  booktitle = {The IEEE/CVF Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}
```
