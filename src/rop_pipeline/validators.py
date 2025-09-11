import numpy as np
import cv2
from PIL import Image
from typing import Union


def is_probable_fundus(img: Union[Image.Image, np.ndarray]) -> bool:
    """
    Fast heuristic to check if an image is likely a retinal fundus photo.
    
    Args:
        img: PIL Image or numpy array (H, W, C) or (H, W)
        
    Returns:
        bool: True if image appears to be a fundus photo, False otherwise
    """
    try:
        # Convert to numpy array and ensure RGB
        if isinstance(img, Image.Image):
            # Convert PIL to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img)
        else:
            img_array = np.array(img)
            
        # Handle grayscale images
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]  # Remove alpha channel
        elif img_array.shape[2] == 3:
            pass  # Already RGB
        else:
            return False
            
        # Convert to grayscale for circle detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min(gray.shape) // 4,  # Minimum distance between circle centers
            param1=50,  # Upper threshold for edge detection
            param2=30,  # Accumulator threshold for center detection
            minRadius=min(gray.shape) // 8,  # Minimum circle radius
            maxRadius=min(gray.shape) // 2   # Maximum circle radius
        )
        
        # Check if we found a large circle (fundus FOV)
        has_large_circle = False
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Check if circle is reasonably large relative to image size
                circle_area_ratio = (np.pi * r * r) / (gray.shape[0] * gray.shape[1])
                if circle_area_ratio > 0.1:  # Circle should occupy at least 10% of image
                    has_large_circle = True
                    break
        
        # Check HSV saturation to avoid grayscale images
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        sat_mean = np.mean(saturation)
        sat_std = np.std(saturation)
        
        # Fundus images typically have some color variation
        # Very low saturation suggests grayscale or poor quality
        has_color_variation = sat_mean > 20 and sat_std > 10
        
        return has_large_circle and has_color_variation
        
    except Exception:
        # If any error occurs during processing, assume it's not a fundus
        return False


def assert_fundus_or_raise(img: Union[Image.Image, np.ndarray]) -> None:
    """
    Assert that an image is a retinal fundus photo, raise ValueError if not.
    
    Args:
        img: PIL Image or numpy array
        
    Raises:
        ValueError: If input is not a retinal fundus photo
    """
    if not is_probable_fundus(img):
        raise ValueError("Input is not a retinal fundus photo (fundus FOV not detected).")


# Export functions
__all__ = ["is_probable_fundus", "assert_fundus_or_raise"]
