## ParamISP: Learned Forward and Inverse ISPs using Camera Parameters<br><sub>Official PyTorch Implementation of the CVPR 2024 Paper</sub>

Woohyeok Kim\*, Geonu Kim\*, Junyong Lee, Seungyong Lee, Seung-Hwan Baek, Sunghyun Cho<br>

[\[Paper\]]()
[\[Supple\]]()

### Environment Setting
* Python 3.10
* PyTorch 1.12.1

      pip install -r requirements.txt

### Training
As described in the paper, ParamISP is trained in two stages for both the inverse and forward directions: pre-training and fine-tuning. Additionally, before applying it to applications, further joint fine-tuning can be conducted. We provide a small dataset example and the official weights reported in the paper to enable the execution of the code.

[[Dataset examples]]() [[Official weights]]()
#### 1. Pre-training

        CUDA_VISIBLE_DEVICES=0 python models/paramisp.py -o demo --inverse train

#### 2. Fine-tuning
        
        CUDA_VISIBLE_DEVICES=0 python models/paramisp.py -o demo --inverse train --camera D7000

#### 3. Joint fine-tuning

        CUDA_VISIBLE_DEVICES=0,1 python models/paramisp_joint.py -o demo train --camera D7000 --pisp-inv weights/fine_tuning/inverse/D7000.ckpt --pisp-fwd weights/fine_tuning/forward/D7000.ckpt



### Test
 we ~
 
        CUDA_VISIBLE_DEVICES=0 python models/paramisp.py -o demo --inverse test --ckpt weights/fine_tuning/inverse/D7000.ckpt --camera D7000

### Inference
 we ~
 
        CUDA_VISIBLE_DEVICES=0 python models/paramisp.py -o demo --inverse predict --ckpt weights/fine_tuning/inverse/D7000.ckpt --camera D7000
