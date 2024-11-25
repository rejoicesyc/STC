# Towards End-to-End Unsupervised Saliency Detection with Self-Supervised Top-Down Context

## Requirements

- Python 3.6
- Pytorch 1.7
- Install submodule `Connected_components_PyTorch`
- Prepare the pretrained Moco-v2 [weight](https://github.com/CVI-SZU/CCAM/blob/master/WSSS/README.md)

## Usage

``` Python
# For training
python train.py    

# For testing a saved model weight from a specific epoch
python test_tool.py  
```

## Acknowledgement

Our idea is inspired by [PSOD](https://github.com/shuyonggao/PSOD), [A2S-v2](https://github.com/moothes/A2S-v2), [C2AM](https://github.com/CVI-SZU/CCAM), and [AFA](https://github.com/rulixiang/afa). We also thank [Connected_components_PyTorch](https://github.com/zsef123/Connected_components_PyTorch) for providing a high-performance algorithm and implementation for calculating connected components.

## Citation

```
@inproceedings{song2023towards,
  title={Towards End-to-End Unsupervised Saliency Detection with Self-Supervised Top-Down Context},
  author={Song, Yicheng and Gao, Shuyong and Xing, Haozhe and Cheng, Yiting and Wang, Yan and Zhang, Wenqiang},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={5532--5541},
  year={2023}
}
```