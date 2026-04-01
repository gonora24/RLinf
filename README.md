<div align="center">
  <img src="https://github.com/RLinf/misc/raw/main/pic/logo_white.svg" alt="RLinf-logo" width="600"/>
</div>

<div align="center">
<a href="https://arxiv.org/abs/2509.15965"><img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv"></a>
<a href="https://huggingface.co/RLinf"><img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
<a href="https://rlinf.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&logo=readthedocs"></a>
<a href="https://rlinf.readthedocs.io/zh-cn/latest/"><img src="https://img.shields.io/badge/中文文档-red?logo=readthedocs"></a>
<a href="https://deepwiki.com/RLinf/RLinf"><img src="https://img.shields.io/badge/Ask%20DeepWiki-1DA1F2?logo=databricks&logoColor=white&color=00ADEF" alt="Ask DeepWiki"></a>
<a href="https://github.com/RLinf/misc/blob/main/pic/wechat.jpg?raw=true"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>
</div>

<div align="center">

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![简体中文](https://img.shields.io/badge/语言-简体中文-red.svg)](README.zh-CN.md)

</div>

<h1 align="center">
  <sub>RLinf: Reinforcement Learning Infrastructure for Embodied and Agentic AI</sub>
</h1>

RLinf is a flexible and scalable open-source RL infrastructure designed for Embodied and Agentic AI. The 'inf' in RLinf stands for `Infrastructure`, highlighting its role as a robust backbone for next-generation training. It also stands for `Infinite`, symbolizing the system’s support for open-ended learning, continuous generalization, and limitless possibilities in intelligence development.

<div align="center">
  <img src="https://github.com/RLinf/misc/raw/main/pic/overview.svg" alt="RLinf-overview"/>
</div>


## Installation

### 1. Clone Git Repo

```
git clone git@github.com:gonora24/RLinf.git
cd RLinf
```
-> use dsrl branch

### 2. Install Virtual Environment

``` 
bash requirements/install.sh embodied --model openpi --env maniskill_libero --venv openpi-venv --no-root 
source openpi-venv/bin/activate 
```

### 3. Download Models

``` # Download the Spatial-Object-Goal model (choose either method)
# Method 1: Using git clone
git lfs install
git clone https://huggingface.co/RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT
git clone https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT

# Method 2: Using huggingface-hub
# For mainland China users, you can use the following for better download speed:
# export HF_ENDPOINT=https://hf-mirror.com
pip install huggingface-hub
hf download RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT --local-dir RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT
hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir RLinf-Pi05-LIBERO-SFT 
```

## Run Training

1. Edit `train_embodiment.sh`
2. ```bash train_embodiment.sh```

## Citation and Acknowledgement

If you find **RLinf** helpful, please cite the paper:

```bibtex
@article{yu2025rlinf,
  title={RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation},
  author={Yu, Chao and Wang, Yuanqing and Guo, Zhen and Lin, Hao and Xu, Si and Zang, Hongzhi and Zhang, Quanlu and Wu, Yongji and Zhu, Chunyang and Hu, Junhao and others},
  journal={arXiv preprint arXiv:2509.15965},
  year={2025}
}
```

If you use RL+VLA in RLinf, you can also cite our technical report and empirical study paper:

```bibtex
@article{zang2025rlinf,
  title={RLinf-VLA: A Unified and Efficient Framework for VLA+ RL Training},
  author={Zang, Hongzhi and Wei, Mingjie and Xu, Si and Wu, Yongji and Guo, Zhen and Wang, Yuanqing and Lin, Hao and Shi, Liangzhi and Xie, Yuqing and Xu, Zhexuan and others},
  journal={arXiv preprint arXiv:2510.06710},
  year={2025}
}
```

```bibtex
@article{liu2025can,
  title={What can rl bring to vla generalization? an empirical study},
  author={Liu, Jijia and Gao, Feng and Wei, Bingwen and Chen, Xinlei and Liao, Qingmin and Wu, Yi and Yu, Chao and Wang, Yu},
  journal={arXiv preprint arXiv:2505.19789},
  year={2025}
}
```

```bibtex
@article{chen2025pi_,
  title={$$\backslash$pi\_$\backslash$texttt $\{$RL$\}$ $: Online RL Fine-tuning for Flow-based Vision-Language-Action Models},
  author={Chen, Kang and Liu, Zhihao and Zhang, Tonghe and Guo, Zhen and Xu, Si and Lin, Hao and Zang, Hongzhi and Zhang, Quanlu and Yu, Zhaofei and Fan, Guoliang and others},
  journal={arXiv preprint arXiv:2510.25889},
  year={2025}
}
```

If you train your policies in physical world with RLinf, you can cite our paper:
```bibtex
@article{zang2026rlinfuser,
  title={RLinf-USER: A Unified and Extensible System for Real-World Online Policy Learning in Embodied AI}, 
  author={Hongzhi Zang and Shu'ang Yu and Hao Lin and Tianxing Zhou and Zefang Huang and Zhen Guo and Xin Xu and Jiakai Zhou and Yuze Sheng and Shizhe Zhang and Feng Gao and Wenhao Tang and Yufeng Yue and Quanlu Zhang and Xinlei Chen and Chao Yu and Yu Wang},
  year={2026},
  journal={arXiv preprint arXiv:2602.07837},
  url={https://arxiv.org/abs/2602.07837}, 
}
```

If you use World Model + VLA + RL in RLinf, you can cite our paper:
```bibtex
@article{jiang2026wovr,
  title={WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL}, 
  author={Zhennan Jiang and Shangqing Zhou and Yutong Jiang and Zefang Huang and Mingjie Wei and Yuhui Chen and Tianxing Zhou and Zhen Guo and Hao Lin and Quanlu Zhang and Yu Wang and Haoran Li and Chao Yu and Dongbin Zhao},
  year={2026},
  journal={arXiv preprint arXiv:2602.13977},
  url={https://arxiv.org/abs/2602.13977}, 
}
```

If you use RL-based sim-real co-training in RLinf, you can cite our paper:
```bibtex
@article{shi2026rlinf,
  title={Beyond Imitation: Reinforcement Learning-Based Sim-Real Co-Training for VLA Models},
  author={Shi, Liangzhi and Chen, Shuaihang and Gao, Feng and Chen, Yinuo and Chen, Kang and Zhang, Tonghe and Zhang, Hongzhi and Zhang, Weinan and Yu, Chao and Wang, Yu},
  journal={arXiv preprint arXiv:2602.12628},
  year={2026},
  url={https://arxiv.org/abs/2602.12628},
}
```

If you use WideSeek-R1 in RLinf, you can cite our paper:
```bibtex
@article{xu2026wideseek,
  title={WideSeek-R1: Exploring Width Scaling for Broad Information Seeking via Multi-Agent Reinforcement Learning},
  author={Xu, Zelai and Xu, Zhexuan and Zhang, Ruize and Zhu, Chunyang and Yu, Shi and Liu, Weilin and Zhang, Quanlu and Ding, Wenbo and Yu, Chao and Wang, Yu},
  journal={arXiv preprint arXiv:2602.04634},
  year={2026},
}
```   

**Acknowledgements**
RLinf has been inspired by, and benefits from, the ideas and tooling of the broader open-source community.
In particular, we would like to thank the teams and contributors behind VeRL, AReaL, Megatron-LM, SGLang, and PyTorch Fully Sharded Data Parallel (FSDP), and if we have inadvertently missed your project or contribution, please open an issue or a pull request so we can properly credit you.

**Contact:**
We welcome applications from Postdocs, PhD/Master's students, and interns. Join us in shaping the future of RL infrastructure and embodied AI!
- Chao Yu: zoeyuchao@gmail.com
- Yu Wang: yu-wang@tsinghua.edu.cn
