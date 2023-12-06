import os
import json
import argparse

import cv2
import matplotlib.pyplot as plt

from docscan.scan import DocScanner, DOCSCAN_OUTPUT_DIR
from doctr.models import ocr_predictor
from doctr.io.reader import DocumentFile, OCR_OUTPUT_DIR

if __name__ == '__main__':
  argparser = argparse.ArgumentParser(description=__doc__)
  argparser.add_argument('--img', type=str, help='Path of image to be scanned')
  args = argparser.parse_args()

  ##### Preprocessing #####
  print('\nPreprocessing...')
  img_path = args.img
  img_name = os.path.splitext(os.path.basename(img_path))[0]
  scanner = DocScanner()
  scanner.scan(img_path)
  scanner.visualize()

  ##### Text detection & recognition #####
  print('\nOCR text detection & recognition...')
  model = ocr_predictor('db_resnet50', 'crnn_mobilenet_v3_large', pretrained=True, assume_straight_pages=True)
  doc = DocumentFile.from_images(os.path.join(DOCSCAN_OUTPUT_DIR, os.path.basename(img_path)))
  result = model(doc)
  result.show()
  # JSON result
  JSON_DIR = os.path.join(os.getcwd(), 'result', 'json')
  json_result = result.export()
  if not os.path.exists(JSON_DIR):
    os.makedirs(JSON_DIR)
  with open(os.path.join(JSON_DIR, img_name + '.json'), 'w') as f:
    json.dump(json_result, f)

  ##### Synthesize text detection & recognition results #####
  print('\nSynthesizing...')
  synthetic_page = result.synthesize()[0]
  cv2.imwrite(os.path.join(OCR_OUTPUT_DIR, os.path.basename(img_path)), synthetic_page)
  plt.figure(); plt.imshow(synthetic_page); plt.title('Synthesized Result'); plt.axis('off'); 

  print('\nPlotting final results...')
  plt.show()