# TaCoMoE_reproduce
Here are the software and hardware environments I used in my experiments:<br>
• Software:<br>
&nbsp;&nbsp;&nbsp;&nbsp;• Python: 3.9.0<br>
&nbsp;&nbsp;&nbsp;&nbsp;• Pytorch: 2.2.0+cu121<br>
&nbsp;&nbsp;&nbsp;&nbsp;• transformers: 4.42.0(qwen2)<br>
&nbsp;&nbsp;&nbsp;&nbsp;• deepspeed: 0.13.1<br>
• Hardware:<br>
&nbsp;&nbsp;&nbsp;&nbsp;• GPU: NVIDIA GeForce RTX 4090<br>
&nbsp;&nbsp;&nbsp;&nbsp;• CUDA: 12.1<br>

# Data & Model<br>
You can download the dataset used in this project from the following link: [Download Dataset](https://github.com/unikcc/DiaASQ).<br>
We provide the implementation for ChatGLM3; the other models can be adapted by modifying their model loading code in the same way as for ChatGLM3.<br>
| Model Name |
|----------|
| [ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b)  | 
| [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b)  | 
| [ChatGLM3-6B ](https://huggingface.co/THUDM/chatglm3-6b) |
| [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B-Instruct)  |

# Running TaCoMoE
After setting up the environment, run the bash script in the 'experiments' directory.
# Experimental Results
| Model(ChatGLM) | T | A | O | T-A | T-O | A-O | Q(P) | Q(R) | Q(F1) |
| ------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Epoch1 | 86.0645 | 70.9576 | 52.5676 | 41.9151 | 31.4843 | 42.3901 | 42.3977 | 26.6055 | 32.6945 |
| Epoch2 | 87.0432 | 69.1865 | 50.0000 | 43.3980 | 37.1831 | 42.8428 | 36.0082 | 32.1101 | 33.9476 |
| Epoch3 | 88.4641 | 70.7951 | 51.9308 | 44.1513 | 40.8078 | 44.4219 | 34.0426 | 38.1651 | 35.9862 |
| Epoch4 | 87.6893 | 72.2267 | 54.1029 | 46.4627 | 40.5680 | 48.3971 | 39.8058 | 37.6147 | 38.6792 |
| Epoch5 | 88.1128 | 71.3608 | 55.0569 | 46.7128 | 40.0509 | 46.2273 | 36.3636 | 35.2294 | 35.7875 |
| Epoch6 | 88.4265 | 70.8745 | 53.3058 | 46.6139 | 39.8746 | 48.4970 | 39.1473 | 37.0642 | 38.0773 |
| Epoch7 | 88.3476 | 71.4961 | 53.5422 | 45.6679 | 40.6892 | 49.3069 | 35.5818 | 38.7156 | 37.0826 |
| Epoch8 | 88.1156 | 71.0731 | 55.3360 | 46.0724 | 41.6141 | 47.1754 | 37.0909 | 37.4312 | 37.2603 |
| Epoch9 | 88.8016 | 71.5180 | 56.0554 | 47.6881 | 42.2553 | 47.6876 | 38.6275 | 36.1468 | 37.3460 |
| Epoch10 | 88.6854 | 70.6719 | 55.3633 | 47.6534 | 40.8797 | 47.9188 | 38.7405 | 37.2477 | 37.9794 |
| **TaCoMoE (ChatGLM3)** | **91.04** | **77.02** | **63.13** | **54.53** | **52.86** | **53.71** | **41.99** | **42.16** | **42.08** |

| Model(Qwen) | T | A | O | T-A | T-O | A-O | Q(P) | Q(R) | Q(F1) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Epoch1 | 86.8909 | 69.1091 | 53.5232 | 41.8811 | 40.2560 | 43.6735 | 38.5305 | 39.4495 | 38.9846 |
| Epoch2 | 88.4668 | 71.6695 | 52.1260 | 47.0904 | 44.3816 | 49.5238 | 42.0290 | 31.9266 | 36.2878 |
| Epoch3 | 89.0220 | 72.5203 | 53.6621 | 48.3146 | 42.0239 | 48.7856 | 45.9584 | 36.5138 | 40.6953 |
| Epoch4 | 88.9764 | 71.5686 | 53.8405 | 48.6017 | 43.6714 | 48.9926 | 46.3470 | 37.2477 | 41.3021 |
| Epoch5 | 88.9914 | 72.2930 | 53.9705 | 47.8617 | 42.2809 | 50.9845 | 42.5577 | 37.2477 | 39.7260 |
| Epoch6 | 88.9182 | 72.6693 | 53.9817 | 49.6377 | 44.3978 | 51.7134 | 41.8557 | 37.2477 | 39.4175 |
| Epoch7 | 88.9182 | 72.6693 | 53.9817 | 49.6377 | 44.3978 | 51.7134 | 41.8557 | 37.2477 | 39.4175 |
| Epoch8 | 88.9182 | 72.6693 | 53.9817 | 49.6377 | 44.3978 | 51.7134 | 41.8557 | 37.2477 | 39.4175 |
| Epoch9 | 88.9182 | 72.6693 | 53.9817 | 49.6377 | 44.3978 | 51.7134 | 41.8557 | 37.2477 | 39.4175 |
| **TaCoMoE (Qwen2-7b)** | **91.26** | **77.81** | **64.34** | **57.42** | **55.39** | **56.5** | **45.63** | **45.87** | **45.75** |