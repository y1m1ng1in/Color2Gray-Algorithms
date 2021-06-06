# Color2Gray-Algorithms

  

Copyright (c) 2021 Yiming Lin

Implementation of color-to-grayscale image conversion algorithms proposed in the following papers:

------------


`spcr.py` implements "Color2Gray: Salience-Preserving Color Removal. Amy A. Gooch, Sven C. Olsen, Jack Tumblin, and Bruce Gooch. SIGGRAPH 05"

Argument:

`--intput`: specify the path to the input color image

`--output`: specify the path to the output grayscale image

`--mu`: specify the width (actual width is 2 * mu) of the squared neighborhood pixel patch

`--npi` and `--dpi`: specify  user parameter `theta` by `npi` times `pi` / `dpi`

`--alpha`: specify user parameter `alpha`


------------


`ngm.py` implements "Robust color-to-gray via nonlinear global mapping. Yongjin Kim and Cheolhun Jang and Julien Demouth and Seungyong Lee. SIGGRAPH ASIA 2009"

Argument:

`--intput`: specify the path to the input color image

`--output`: specify the path to the output grayscale image

`--dof`: degree of trigonometric polynomial used, default is 4 as paper suggested

`--alpha`: specify user parameter `alpha`

`--lamb`: specify `lamb` * number of pixels for `E_r` term, where `E_image = E_s + E_r` as paper suggested


------------


The following commands contains parameters used for output with reasonable quality, where the output image paths are all set to `testarg.png`:
```shell
python .\ngm.py --input ".\images\map.png" --output "testarg.png" -a 0.1 -l 1
python .\ngm.py --input ".\images\sample.png" --output "testarg.png" -a 4 -l 1
python .\ngm.py --input ".\images\colorCircle.png" --output "testarg.png" -a 1 -l 1
python .\ngm.py --input ".\images\dots.png" --output "testarg.png" -a 1 -l 1
python .\ngm.py --input ".\images\MonetP.png" --output "testarg.png" -a 4 -l 1
python .\ngm.py --input ".\images\lischinski.png" --output "testarg.png" -a 0.1 -l 500

python .\spcr.py --input ".\images\sample_30.png" --output "testarg.png" --mu 8
python .\spcr.py --input ".\images\dots.png" --output "testarg.png" --mu 16
python .\spcr.py --input ".\images\mapIslandCrop.png" --output "testarg.png" --mu 16 --npi 8
python .\spcr.py --input ".\images\MonetP.png" --output "testarg.png" --mu 16
python .\spcr.py --input ".\images\lischinski.png" --output "testarg.png" --mu 32 --npi 2 --alpha 10


```
