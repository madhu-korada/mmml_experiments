import cv2
import numpy as np

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
    else:
        return image

def adjust_font_properties(img_width, text, base_font_size=1.0, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    Adjusts font size and thickness based on the length of the text and the image width.
    """
    scale_factor = img_width / 500  # Adjust scale factor based on image width
    base_font_size *= scale_factor
    (text_width, _), _ = cv2.getTextSize(text, font, base_font_size, 1)
    available_space = img_width * 0.8  # Use up to 80% of the width for text
    if text_width > available_space:
        font_size = base_font_size * (available_space / text_width)
    else:
        font_size = base_font_size
    thickness = max(1, int(font_size / 10))  # Adjust thickness based on font size
    return font_size, thickness

def adjust_font_size(img_width, text, base_font_size=1.0, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    Adjusts the font size based on the length of the text and the image width.
    """
    (text_width, _), _ = cv2.getTextSize(text, font, base_font_size, 1)
    available_space = img_width * 0.8  # Assuming we want to use up to 80% of the width for text
    if text_width > available_space:
        return base_font_size * (available_space / text_width)
    return base_font_size

def text_on_image(image, mod_image_path='modified_image.jpg', line1_text="Line 1 of Text", line2_text="Line 2 of Text"):
    # Step 1: Calculate the border size (20% of the image's width)
    border_size = int(0.2 * image.shape[1])
    top_border = np.full((border_size, image.shape[1], 3), 255, dtype=np.uint8)  # White border

    # Step 2: Attach the white border to the top of the image
    bordered_image = cv2.vconcat([top_border, image])

    # Step 3: Add text to the image
    # Adjust font size based on text length
    font_size1, thickness1 = adjust_font_properties(image.shape[1], line1_text)
    font_size2, thickness2 = adjust_font_properties(image.shape[1], line2_text)
    
    # Add text to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (0, 0, 20)  # Black color
    cv2.putText(bordered_image, line1_text, (10, int(border_size / 2) - 10), font, font_size1, text_color, thickness1)
    cv2.putText(bordered_image, line2_text, (10, int(border_size / 2) + 30), font, font_size2, text_color, thickness2)

    # Step 4: Save the modified image
    cv2.imwrite(mod_image_path, bordered_image)

    print("Image saved as:", mod_image_path)                   
    return image

def draw_text_on_image_given_path(image_path, mod_image_path, line1_text, line2_text):
    image = load_image(image_path)
    modified_image = text_on_image(image, mod_image_path, line1_text, line2_text)
    return modified_image

if __name__ == "__main__":
    image_path = 'data/coco2014/val2014/COCO_val2014_000000099647.jpg'  # Update this to your image's path
    mod_image_path = 'modified_image.jpg'
    
    modified_image = draw_text_on_image_given_path(image_path, mod_image_path, "Question: Is this tennis player playing tennis or just being dramatic?", "Gold: ['play', 'play', 'play', 'play', 'play', 'play', 'play', 'play', 'game', 'game']")
    
    # image = load_image(image_path)
    # modified_image = text_on_image(image, mod_image_path, "Question: Is this tennis player playing tennis or just being dramatic?", "Gold: ['play', 'play', 'play', 'play', 'play', 'play', 'play', 'play', 'game', 'game']")
    