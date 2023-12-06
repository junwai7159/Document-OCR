import os
import math
import itertools

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist

from pylsd.lsd import lsd
from .utils import imgutils

fig1, axs1 = plt.subplots(2, 2)
fig2, axs2 = plt.subplots(2, 2)
fig3, axs3 = plt.subplots(2, 3)

# Constants
RESIZE_HEIGHT = 500.0
MIN_QUAD_AREA_RATIO = 0.25
MAX_QUAD_ANGLE_RANGE = 40
CORNERS_MIN_DIST = 20
MORPH = 9
CANNY = 84
DOCSCAN_OUTPUT_DIR = os.path.join(os.getcwd(), 'result', 'preprocess_img')

class DocScanner():
    """A class used to scan documents from images"""

    def __init__(self, min_quad_area_ratio=MIN_QUAD_AREA_RATIO, min_quad_angle_range=MAX_QUAD_ANGLE_RANGE):
        """
        Initializes the Scan object.

        Args:
            min_quad_area_ratio (float): The minimum ratio of the area of a contour's quadrilateral
                to the area of the original image. A contour will be rejected if its corners do not
                form a quadrilateral that covers at least this ratio of the original image. Defaults to 0.25.
            min_quad_angle_range (int): The maximum range of interior angles allowed for a contour's
                quadrilateral. A contour will be rejected if the range of its interior angles exceeds
                this value. Defaults to 40.
        """
        self.min_quad_area_ratio = min_quad_area_ratio
        self.min_quad_angle_range = min_quad_angle_range
        
    def filter_corners(self, corners, corners_min_dist=CORNERS_MIN_DIST):
        """
        Filters corners that are within a minimum distance of each other.

        Args:
            corners (list): A list of corner coordinates.
            corners_min_dist (int, optional): The minimum distance between corners. Defaults to 20.

        Returns:
            list: A list of filtered corner coordinates.

        """
        def predicate(representatives, corner):
            return all(dist.euclidean(representative, corner) >= corners_min_dist
                       for representative in representatives)

        filtered_corners = []
        for c in corners:
            if predicate(filtered_corners, c):
                filtered_corners.append(c)
        return filtered_corners

    def calc_2vector_angle(self, u, v):
        """
        Calculates the angle in degrees between two vectors.

        Args:
            u (numpy.ndarray): The first vector.
            v (numpy.ndarray): The second vector.

        Return:
        Args:
            u (numpy.ndarray): The first vector.
            v (numpy.ndarray): The second vector.

        Returns:
            float: The angle between the two vectors in degrees.
        """
        return np.degrees(math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))

    def get_angle(self, p1, p2, p3):
        """
        Returns the angle between the line segment from p2 to p1 
        and the line segment from p2 to p3 in degrees.

        Args:
        - p1: The first point (x, y) of the line segment.
        - p2: The second point (x, y) of the line segment. The angle is measured from this point.
        - p3: The third point (x, y) of the line segment.

        Returns:
        - angle: The angle between the line segments in degrees.
        """
        a = np.radians(np.array(p1))
        b = np.radians(np.array(p2))
        c = np.radians(np.array(p3))

        avec = a - b
        cvec = c - b

        return self.calc_2vector_angle(avec, cvec)

    def get_quad_angle_range(self, quad):
        """
        Returns the range between the maximum and minimum interior angles of a quadrilateral.

        Args:
        quad (numpy.ndarray): The input quadrilateral as a numpy array with vertices ordered clockwise
                              starting with the top left vertex.

        Returns:
        float: The range between the maximum and minimum interior angles of the quadrilateral.
        """

        tl, tr, br, bl = quad
        ura = self.get_angle(tl[0], tr[0], br[0])
        ula = self.get_angle(bl[0], tl[0], tr[0])
        lra = self.get_angle(tr[0], br[0], bl[0])
        lla = self.get_angle(br[0], bl[0], tl[0])

        angles = [ura, ula, lra, lla]
        return np.ptp(angles)

    def get_corners(self, img):
        """
        Returns a list of corners ((x, y) tuples) found in the input image. With proper
        pre-processing and filtering, it should output at most 10 potential corners.
        
        Parameters:
            img (numpy.ndarray): The input image, expected to be rescaled and Canny filtered.
            
        Returns:
            list: A list of corners as (x, y) tuples.
            
        Notes:
            This is a utility function used by get_contours. The input image is expected 
            to be rescaled and Canny filtered prior to being passed in.
        """

        # LSD - Line Segment Detector
        # returns N*5 numpy.array
        # 5-dimensional vector: [point1.x, point1.y, point2.x, point2.y, width]
        lines = lsd(img)

        corners = []
        if lines is not None:
            lines = lines.squeeze().astype(np.int32).tolist()

            # blank canvas to draw lines on
            lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)

            for line in lines:
                x1, y1, x2, y2, _ = line
                axs3[0,0].imshow(cv2.line(lines_canvas, (x1, y1), (x2, y2), 255, 2), cmap='gray')

                # find horizontal lines
                if abs(x2 - x1) > abs(y2 - y1):
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
                    axs3[0,1].imshow(cv2.line(horizontal_lines_canvas, (max(x1 - 5, 0), y1), (min(x2 + 5, img.shape[1] - 1), y2), 255, 2), cmap='gray')
                # find vertical lines
                else:
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1])
                    axs3[0,2].imshow(cv2.line(vertical_lines_canvas, (x1, max(y1 - 5, 0)), (x2, min(y2 + 5, img.shape[0] - 1)), 255, 2), cmap='gray')
            lines = []
            
            # find 2 horizontal lines of the document
            # connected-components -> bounding boxes -> final lines
            (contours, hierarchy) = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_x = np.amin(contour[:, 0], axis=0) + 2
                max_x = np.amax(contour[:, 0], axis=0) - 2
                left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
                right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
                lines.append((min_x, left_y, max_x, right_y))
                hl = cv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 1, 1)
                axs3[1,0].imshow(hl, cmap='gray')
                corners.append((min_x, left_y))
                corners.append((max_x, right_y))

            # find 2 vertical lines of the document
            # connected-components -> bounding boxes -> final lines
            (contours, hierarchy) = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_y = np.amin(contour[:, 1], axis=0) + 2
                max_y = np.amax(contour[:, 1], axis=0) - 2
                top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
                bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
                lines.append((top_x, min_y, bottom_x, max_y))
                vl = cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 1, 1)
                axs3[1,1].imshow(vl, cmap='gray')
                corners.append((top_x, min_y))
                corners.append((bottom_x, max_y))

            # find the corners
            corners_y, corners_x = np.where(horizontal_lines_canvas + vertical_lines_canvas == 2)
            axs3[1,2].imshow(horizontal_lines_canvas + vertical_lines_canvas, cmap='gray')
            corners += zip(corners_x, corners_y)

        # filter out corners in close proximity
        corners = self.filter_corners(corners)
        return corners

    def is_valid_contour(self, cnt, IMG_WIDTH, IMG_HEIGHT):
        """
        Checks if the given contour is a valid quadrilateral based on the specified requirements.

        Args:
            cnt (numpy.ndarray): The contour to be checked.
            IMG_WIDTH (int): The width of the image.
            IMG_HEIGHT (int): The height of the image.

        Returns:
            bool: True if the contour satisfies all requirements, False otherwise.
        """
        return (len(cnt) == 4 and cv2.contourArea(cnt) > IMG_WIDTH * IMG_HEIGHT * self.min_quad_area_ratio and self.get_quad_angle_range(cnt) < self.min_quad_angle_range)

    def find_contour(self, resize_img):
        """
        Finds the contour of a document in the given resized image and returns the vertices of the four corners of the document as a numpy array of shape (4, 2). 
        If no valid contour is found, the method assumes the entire image is the document and returns the four corners of the image as the contour.

        Parameters:
        - resize_img: A numpy array representing the resized image.

        Returns:
        - cnt_pts: A numpy array of shape (4, 2) containing the vertices of the four corners of the document.
        """

        IMG_HEIGHT, IMG_WIDTH, _ = resize_img.shape

        # convert to grayscale and blur to smooth noise
        gray = cv2.cvtColor(resize_img, cv2.COLOR_RGB2GRAY)
        axs2[0,0].imshow(gray, cmap='gray')
        gray = cv2.GaussianBlur(gray, (7,7), 0)
        axs2[0,1].imshow(gray, cmap='gray')

        # remove potential holes between edge segments gaps using morphological transformation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH,MORPH))
        dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        axs2[1,0].imshow(dilated, cmap='gray')

        # find edges using Canny edge detector
        edged = cv2.Canny(dilated, 0, CANNY)
        axs2[1,1].imshow(edged, cmap='gray')
        test_corners = self.get_corners(edged)
        approx_contours = []

        if len(test_corners) >= 4:
            quads = []
            for quad in itertools.combinations(test_corners, 4):
                points = np.array(quad)
                points = imgutils.order_points(points)
                points = np.array([[p] for p in points], dtype = "int32")
                quads.append(points)
            # get top five quadrilaterals by area
            quads = sorted(quads, key=cv2.contourArea, reverse=True)[:5]
            # sort candidate quadrilaterals by their angle range, which helps remove outliers
            quads = sorted(quads, key=self.get_quad_angle_range)

            approx = quads[0]
            if self.is_valid_contour(approx, IMG_WIDTH, IMG_HEIGHT):
                approx_contours.append(approx)

            # draw the corners and countour found 
            cv2.drawContours(resize_img, [approx], -1, (20, 20, 255), 2)
            axs1[0,1].scatter(*zip(*test_corners))
            axs1[0,1].imshow(resize_img)

        # use the whole image, if no valid contour is found
        if not approx_contours:
            TOP_RIGHT = (IMG_WIDTH, 0)
            BOTTOM_RIGHT = (IMG_WIDTH, IMG_HEIGHT)
            BOTTOM_LEFT = (0, IMG_HEIGHT)
            TOP_LEFT = (0, 0)
            cnt_pts = np.array([[TOP_RIGHT], [BOTTOM_RIGHT], [BOTTOM_LEFT], [TOP_LEFT]])
        else:
            cnt_pts = max(approx_contours, key=cv2.contourArea)

        return cnt_pts.reshape(4, 2)

    def scan(self, img_path):
        """
        Perform document scanning on the given image.

        Args:
            img_path (str): The path to the input image file.

        Returns:
            None

        Raises:
            None
        """
    
        # load and resize image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs1[0,0].imshow(img)

        orig = img.copy()
        resize_img = imgutils.resize(img, height = int(RESIZE_HEIGHT))

        # find contour
        cnt_pts = self.find_contour(resize_img)

        # perspective transformation
        warped = imgutils.four_point_transform(orig, cnt_pts * img.shape[0] / RESIZE_HEIGHT)
        axs1[1,0].imshow(warped)
        warped = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)

        # sharpen the image
        sharpen = cv2.GaussianBlur(warped, (0,0), 3)
        sharpen = cv2.addWeighted(warped, 1.5, sharpen, -0.5, 0)

        # binarization
        thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
        axs1[1,1].imshow(thresh, cmap='gray')

        # save preprocessed image 
        img_name = os.path.basename(img_path)
        if os.path.exists(DOCSCAN_OUTPUT_DIR):
            cv2.imwrite(os.path.join(DOCSCAN_OUTPUT_DIR, img_name), thresh)
            print("Scanned " + img_name)
        else:
            os.mkdir(DOCSCAN_OUTPUT_DIR)
            cv2.imwrite(os.path.join(DOCSCAN_OUTPUT_DIR, img_name), thresh)
            print("Scanned " + img_name)

    def visualize(self):
        """
        Visualizes the different stages of the document scanning process.

        Returns:
            None
        """
        
        # Overall result
        fig1.suptitle('Overall')
        axs1[0,0].set_title('Original Image'); axs1[0,0].axis('off')
        axs1[0,1].set_title('Contour'); axs1[0,1].axis('off')
        axs1[1,0].set_title('Warped Image'); axs1[1,0].axis('off')
        axs1[1,1].set_title('Thresholded Image'); axs1[1,1].axis('off')

        # Edge detection
        fig2.suptitle('Edge Detection')
        axs2[0,0].set_title('Gray'); axs2[0,0].axis('off')
        axs2[0,1].set_title('Gaussian Blur'); axs2[0,1].axis('off')
        axs2[1,0].set_title('Dilated'); axs2[1,0].axis('off')
        axs2[1,1].set_title('Canny'); axs2[1,1].axis('off')

        # Contour detection
        fig3.suptitle('Contour Detection')
        axs3[0,0].set_title('LSD'); axs3[0,0].axis('off')
        axs3[0,1].set_title('Horizontal Lines'); axs3[0,1].axis('off')
        axs3[0,2].set_title('Vertical Lines'); axs3[0,2].axis('off')
        axs3[1,0].set_title('Filtered Horizontal Lines'); axs3[1,0].axis('off')
        axs3[1,1].set_title('Filtered Vertical Lines'); axs3[1,1].axis('off')
        axs3[1,2].set_title('Contour'); axs3[1,2].axis('off')