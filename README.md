# Document OCR
## About the Project
This project is part of the SJTU ICE4309 - Image Processing & Content Analysis course. We implemented an OCR framework for converting in-the-wild documents to digitally readable and recognizable text.

## Features
The model architecture of Document OCR is shown below: 

<div style="text-align: center;">
  <img src="document_ocr.jpg" alt="Model Architecture" width="50%">
</div>

- The images undergo **preprocessing**, including **edge detection, contour detection, perspective transformation and binarization** to further enhance the image.
- The **text detection** module uses the **DBNet** model with **MobileNetV3** as the backbone network.
- The **text recognition** module uses the **CRNN** model with **MobileNetV3** as the backbone network.

## Getting Started
To get started with your project, follow the steps below to set up your environment, install the necessary dependencies.
#### Create and activate new conda environment
```bash
conda create -n ocr python=3.9
conda activate ocr
```

#### Install pip requirements
```bash
pip install -r requirements.txt
```

## Usage
#### Run the script
```bash
python run.py --img <IMG_DIR> --preprocess 
```
Replace `<IMG_DIR>` with the path to a single image. Specify `--preprocess` to preprocess the input image

#### Example
```bash
python run.py --img input_img/receipt.jpg --preprocess
```

## Demonstrations
#### Edge Detection
| Input Image | Grayscale Conversion | Gaussian Blur | Closing | Canny |
| --- | --- | --- | --- | --- |
| ![image](https://github.com/user-attachments/assets/7ae06d47-8feb-4219-9095-5339cff47f91) | ![image](https://github.com/user-attachments/assets/3ed48584-7971-434d-895a-7b6db2744792) | ![image](https://github.com/user-attachments/assets/2ae87c7a-75cb-4732-8e7e-993746f783c0) | ![image](https://github.com/user-attachments/assets/f88f29af-b25d-4927-9ba5-d01ac96d324d) | ![image](https://github.com/user-attachments/assets/dfe1256e-8fbc-475f-8869-a237b37dd6ea) |

#### Contour Detection
| LSD | Horizontal Line Segments | Vertical Line Segments | Final Contour |
| --- | --- | --- | --- | 
| ![image](https://github.com/user-attachments/assets/41054e9f-65dd-4fb3-960b-f3d8f8eb1c4f) | ![image](https://github.com/user-attachments/assets/61199a89-bf2e-4d74-a909-7da408e3c8af) | ![image](https://github.com/user-attachments/assets/6710e79b-d0c3-45c2-9e55-4f27eae37785) | ![image](https://github.com/user-attachments/assets/44eeabb2-d618-4dfc-89cf-a2d78e2d89c8) |

#### Perspective Transformation & Binarization
| Perspective Transformation | Binarization |
| --- | --- |
| ![image](https://github.com/user-attachments/assets/0809b2d2-7bfc-48f2-85dc-53796da5a8ee) | ![image](https://github.com/user-attachments/assets/a3555602-bf4d-4cd9-bdcf-20e69170f44e) |

#### Text Detection & Recognition
| Text Detection | Text Recognition |
| --- | --- |
| ![image](https://github.com/user-attachments/assets/939aa972-f522-4d9c-aa2b-2a2096db28b7) | ![image](https://github.com/user-attachments/assets/ab68c08a-b65a-4049-81ba-80a0e0d48f25) |
