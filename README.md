# DepthEdgeAwareBENet

A. Karaali, N. Harte, CR. Jung, "Deep Multi-Scale Feature Learning for Defocus Blur Estimation", IEEE Transactions on Image Processing (TIP 2022), 2022. 
To read paper, please refer: https://arxiv.org/abs/2009.11939

Any papers using this code should cite the paper accordingly. 
```
@ARTICLE{9673106,
  author={Karaali, Ali and Harte, Naomi and Jung, Claudio R.},
  journal={IEEE Transactions on Image Processing}, 
  title={Deep Multi-Scale Feature Learning for Defocus Blur Estimation}, 
  year={2022},
  volume={31},
  number={},
  pages={1097-1106},
  doi={10.1109/TIP.2021.3139243}}
```


As is, the code produces the results given as the first experimental setting with dataset which is provided in ('Non-parametric blur map regression for depth of field extension'). In order to reach the dataset, you should contact with the author of this paper. The code here includes just 1 sample images from this dataset.

### Installation

The project is managed with [uv](https://docs.astral.sh/uv/) and implemented in PyTorch.

```sh
$ uv sync
```

### Running

```sh
$ uv run benet -i images/image_01.png
```

This writes `image_01_bmap.png` (blur map, scaled so that sigma = 6 maps to 255) and `image_01_edge.png`
(all edge pixels in blue, depth edges additionally in red) next to the input image. Use `-o DIR` to write elsewhere,
and `--device cuda` to run the networks on a GPU.

In order to make a fair comparisons, we use the same edge maps in some other edge based defocus blur estimations. If you want to use the precomputed edges, which are provided in "precomputed_edges/"

```sh
$ uv run benet -i images/image_01.png -e precomputed_edges/edge_01.png
```

### Python API

```python
import numpy as np
from PIL import Image
from benet import estimate_blur_map, load_networks

bnet, enet = load_networks()  # or load_networks(device="cuda")
img = np.asarray(Image.open("images/image_01.png").convert("RGB"))
result = estimate_blur_map(img, bnet, enet)
result.blur_map  # (H, W) float array, Gaussian sigma in pixels
```

### Pretrained weights

The pretrained networks are stored as PyTorch state dicts in `src/benet/weights/` (`bnet.pth`, `enet.pth`)
and can be loaded with plain PyTorch:

```python
import torch
from benet import BNet

state_dict = torch.load("src/benet/weights/bnet.pth", weights_only=True)
bnet = BNet()
bnet.load_state_dict(state_dict)
```

They were converted from the original TensorFlow 1.x exports in `models_tf1x/`:

```sh
$ uv run --group convert scripts/convert_tf_weights.py
```

Please also report any bug to alixkaraali[at_sign]gmail[dot_sign]com


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
