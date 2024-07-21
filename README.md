# Document scanner & OCR
Project - SJTU ICE4309 - Image Processing & Content Analysis

### Usage
#### Create and activate new conda environment
```bash
conda create -n ocr python=3.9
conda activate ocr
```

#### Install pip requirements
```bash
pip install -r requirements.txt
```

#### Run the script
```bash
python run.py --img <IMG_DIR> --preprocess 
```
Replace <IMG_DIR> with the path to a single image
Specify ```--preprocess``` to preprocess the input image

#### Example
```bash
python run.py --img input_img/receipt.jpg --preprocess
```

#### Results