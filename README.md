# Hand recognition

Program uses ResNet ([paper link](https://arxiv.org/abs/1512.03385)) for object classification and selective search ([paper link](https://koen.me/research/pub/uijlings-ijcv2013-draft.pdf)) to find possible object boxes.<br/>
Packages required to run program are in `requirements.txt`. To use GPU install `CUDA v9.0` and `tensorflow-gpu` package should be installed ([instructions](https://www.tensorflow.org/install/)).<br/>
Program should be run from `gui.py`. Result:<br/>
![image](https://i.imgur.com/gnFQmqN.png) <br/>
**Min confidence** (0-100): specifies the limit from which object boxes are accepted <br/>
**Overlap threshold** (0-1): specifies threshold in non-max suppresion algorithm
