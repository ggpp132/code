# git
![til](https://github.com/abzd43643/git/blob/main/demo.gif)
## Dependencies
- Python 3.9
- PyTorch 2.0.1
- NVIDIA GPU + CUDA (https://developer.nvidia.com/cuda-downloads)

## Create environment and install packages
- `conda create -n git python=3.9`
- `conda activate git`
- `pip install -r requirements.txt`

## Testing

- test datasets in location `test_img`.

- Test the MI, VIF, Qabf, SSIM metrics on the six test sets
`python test.py`

- The output images are in `results/`.
 
