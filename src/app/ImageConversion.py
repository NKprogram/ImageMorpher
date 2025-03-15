# Standard Library Imports
import random
from math import pi
import colorsys

# Third-Party Libraries Imports
import cv2
import numpy as np
from PIL import (
    Image,
    ImageDraw,
    ImageFont,
    ImageFilter,
    ImageOps,
    ImageEnhance,
)
import PIL.ImageChops as ImageChops
from pilmoji import Pilmoji
from glitch_this import ImageGlitcher
from matplotlib import pyplot as plt
from skimage import draw


def convert_image_to_ascii(image_path: str, output_path="converted_ascii.png") -> str:
    # are the characters used to represent the image, from darkest to lightest
    ASCII_CHARS = list("@%#*+=-:.")
    #open the image
    img = Image.open(image_path)
    #resizing the image to the standard discord size
    max_width = 80
    orig_w, orig_h = img.size
    if orig_w > max_width:
        aspect = orig_h / orig_w
        new_w = max_width
        new_h = int(aspect * new_w * 0.55)
        img = img.resize((new_w, new_h))
    #convert it to grayscale   
    img = img.convert("L")
    w,h = img.size
    pixels = img.getdata()
    ascii_array = []
    for p in pixels:
        idx = p * len(ASCII_CHARS) // 256
        ascii_array.append(ASCII_CHARS[idx])
    # Break pixels into lines
    lines = []
    for row_start in range(0, len(ascii_array), w):
        row_chars = ascii_array[row_start : row_start + w]
        line = "".join(row_chars)
        lines.append(line)
    # monospaced font
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"  
    font_size = 14
    font = ImageFont.truetype(font_path, font_size)
    # Measure text size
    sample_text = "@" * w  # represent one full line
    mask = font.getmask(sample_text)
    line_width, line_height = mask.size  
    # total canvas size = line_height * number_of_lines
    total_width = line_width
    total_height = line_height * len(lines)
    # Create the result image
    result_img = Image.new("RGB", (total_width, total_height), "white")
    draw_result = ImageDraw.Draw(result_img)
    # Render lines
    y_offset = 0
    for line in lines:
        draw_result.text((0, y_offset), line, fill="black", font=font)
        y_offset += line_height
    # Save and return
    result_img.save(output_path)
    return output_path

def convert_image_to_emoji(image_path: str, output_path="converted_emoji.png") -> str:
    EMOJI_CHARS = [
        '🌑', '🕳️', '🖤', '🏴',  # very dark
        '🌒', '🌘', '🌚', '🐺', '🐍', '🦇', '🐢', '🦉',  # mid-dark to mid-light
        '🌕', '🌞', '⭐', '🌟', '⚡', '🔥', '🐱', '🐶', '🐥', '🐸',  # mostly bright
        '🌝', '🌞', '❄️', '☁️', '🐇', '🐑', '🕊️'  # near-white
    ]
    # Load the image and resize it
    img = Image.open(image_path).convert("L")
    max_width = 150  
    orig_w, orig_h = img.size
    aspect_ratio = orig_h / orig_w
    new_w = min(orig_w, max_width)
    new_h = int(aspect_ratio * new_w * 0.55)
    img = img.resize((new_w, new_h), Image.LANCZOS) 
    w, h = img.size
    # Convert the image to a list of pixel values
    pixels = img.getdata()
    # Convert each pixel to an emoji
    emoji_array = [EMOJI_CHARS[p * len(EMOJI_CHARS) // 256] for p in pixels]
    # Break the emoji into lines
    lines = ["".join(emoji_array[i: i + w]) for i in range(0, len(emoji_array), w)]
    #monospaced font
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    font_size = 14
    font = ImageFont.truetype(font_path, font_size)
    # Measure text size 
    mask = font.getmask("⬛" * w)
    line_width, line_height = int(mask.size[0] * 1.6), int(mask.size[1] * 1.5)
    # Calculate final image size with proper alignment
    line_spacing = 2  
    total_width = line_width
    total_height = (line_height + line_spacing) * len(lines)
    # Create the result image
    result_img = Image.new("RGB", (total_width, total_height), "white")
    # Proper text alignment and drawing
    with Pilmoji(result_img) as pilmoji:
        y_offset = 0
        for line in lines:
            pilmoji.text(
                ((total_width - line_width) // 2, y_offset),  # Centering text
                line,
                font=font,
                fill=(0, 0, 0)
            )
            y_offset += (line_height + line_spacing)
    # Save and return
    result_img.save(output_path)
    return output_path

def convert_image_to_pixel_art(image_path: str, output_path="converted_pixel_art.png") -> str:
    LEGO_PALETTE = [
        (( 30,  30,  30), "⬛"),  # black
        ((255, 255, 255), "⬜"),  # white
        ((255,   0,   0), "🟥"),  # red
        ((  0, 255,   0), "🟩"),  # green
        ((  0,   0, 255), "🟦"),  # blue
        ((255, 255,   0), "🟨"),  # yellow
        ((255, 165,   0), "🟧"),  # orange
        ((128,   0, 128), "🟪"),  # purple
        ((165,  42,  42), "🟫"),  # brown
    ]
    
    # Find the nearest LEGO emoji for a given RGB color
    def nearest_lego_emoji(r, g, b):
        best_dist = float('inf')
        best_emoji = None
        for (cr, cg, cb), emoji in LEGO_PALETTE:
            dist = (r - cr)**2 + (g - cg)**2 + (b - cb)**2
            if dist < best_dist:
                best_dist = dist
                best_emoji = emoji
        return best_emoji

    # 1) Load and resize
    img = Image.open(image_path).convert("RGB")
    max_width = 150
    orig_w, orig_h = img.size
    if orig_w > max_width:
        aspect_ratio = orig_h / orig_w
        new_w = max_width
        new_h = int(aspect_ratio * new_w)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    w, h = img.size
    pixels = img.getdata()
    # Convert each pixel to a LEGO piece
    lego_emojis = []
    for y in range(h):
        row_emojis = []
        for x in range(w):
            r, g, b = pixels[y * w + x]
            row_emojis.append(nearest_lego_emoji(r, g, b))
        lego_emojis.append("".join(row_emojis))
    # render the LEGO image
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    font_size = 14
    font = ImageFont.truetype(font_path, font_size)
    sample_line = max(lego_emojis, key=len)
    mask = font.getmask(sample_line)
    line_width = mask.size[0]
    line_height = mask.size[1]
    # Calculate and resize the final image 
    horizontal_padding_factor = 1.6
    vertical_padding_factor = 1.5
    line_width = int(line_width * horizontal_padding_factor)
    line_height = int(line_height * vertical_padding_factor)
    total_width = line_width
    total_height = line_height * len(lego_emojis)
    result_img = Image.new("RGB", (total_width, total_height), "white")
    # Proper text alignment and drawing
    with Pilmoji(result_img) as pilmoji:
        y_offset = 0
        for line in lego_emojis:
            pilmoji.text(
                (0, y_offset),
                line,
                font=font,
                fill=(0, 0, 0)
            )
            y_offset += line_height
    result_img.save(output_path)
    return output_path

def convert_image_to_blur(image_path: str, output_path="converted_blur.png") -> str:
    img = Image.open(image_path)
    blurred_img = img.filter(ImageFilter.GaussianBlur(10))
    blurred_img.save(output_path)
    return output_path

def convert_image_to_deep_fry(image_path: str, output_path="converted_deep_fry.png") -> str:
    img = Image.open(image_path).convert("RGB")
    # Posterize the image to reduce the number of colors
    poster_bits = 2
    posterized = ImageOps.posterize(img, bits=poster_bits)
    posterized = ImageEnhance.Color(posterized).enhance(2.0)
    posterized = ImageEnhance.Contrast(posterized).enhance(2.0)
    posterized = ImageEnhance.Brightness(posterized).enhance(1.3)
    posterized = ImageEnhance.Sharpness(posterized).enhance(2.0)
    # Find edges in the image
    gray = img.convert("L")
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.invert(edges)
    edges = ImageEnhance.Contrast(edges).enhance(3.0)
    edges = edges.point(lambda x: 255 if x > 90 else 0)
    edges = edges.convert("RGB")
    # Multiply, constrast and sharpen the posterized image with the edges to get the final deep-fried image
    deep_fry_image = ImageChops.multiply(posterized, edges)
    deep_fry_image = ImageEnhance.Contrast(deep_fry_image).enhance(1.5)
    deep_fry_image = ImageEnhance.Sharpness(deep_fry_image).enhance(1.5)
    deep_fry_image.save(output_path)
    return output_path

def convert_image_to_sketch(image_path: str, output_path="converted_sketch.png") -> str:
    img = Image.open(image_path).convert("L")
    # Invert the image
    inverted = ImageOps.invert(img)
    # apply Gaussian Blur 
    blurred = inverted.filter(ImageFilter.GaussianBlur(21))  
    # Convert both images to NumPy arrays
    gray_np = np.array(img, dtype=np.float32)
    blurred_np = np.array(blurred, dtype=np.float32)
    # Avoid division by zero
    epsilon = 1e-6 
    sketch_np = gray_np * 255 / (255 - blurred_np + epsilon)
    # Clip to [0, 255] and convert back to uint8
    sketch_np = np.clip(sketch_np, 0, 255).astype(np.uint8)
    # Convert NumPy array back to a PIL image and save
    sketch_img = Image.fromarray(sketch_np, mode='L')
    sketch_img.save(output_path)
    return output_path

def convert_image_to_oil_paint(image_path: str, output_path: str = "converted_oil_paint.png") -> str:
    input_image = plt.imread(image_path)
    if input_image.ndim < 3:
        raise ValueError("Only RGB or RGBA images are supported.")
    elif input_image.shape[2] == 4:
        input_image = input_image[:, :, :3]
    # random seed 
    random.seed(0)
    # Default parameters
    brush_size = 10.0
    expression_level = 2.0
    BRUSHES = 50

    # Convert brush size and expression level to integers
    brush_size_int = int(brush_size) 
    expression_size = brush_size * expression_level
    margin = int(expression_size * 2)
    half_brush_size_int = brush_size_int // 2

    # Create a list of brushes
    brushes = [
        draw.ellipse(
            half_brush_size_int,
            half_brush_size_int,
            brush_size,
            random.randint(brush_size_int, int(expression_size)),
            rotation=random.random() * pi
        )
        for _ in range(BRUSHES)
    ]
    
    # Create the oil painting effect
    result_image = np.zeros(input_image.shape, dtype=np.uint8)
    for x in range(margin, input_image.shape[0] - margin, brush_size_int):
        for y in range(margin, input_image.shape[1] - margin, brush_size_int):
            ellipse_xs, ellipse_ys = random.choice(brushes)
            result_image[x + ellipse_xs, y + ellipse_ys] = input_image[x, y]
    plt.imsave(output_path, result_image)
    return output_path

def convert_image_to_watercolor(image_path: str, output_path: str = "converted_watercolor.png") -> str:
    # Read and resize image
    image = cv2.imread(image_path)
    scale = float(3000) / (image.shape[0] + image.shape[1])
    image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    # Convert to HSV and apply washout effect
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    adjust_v = (img_hsv[:, :, 2].astype("uint") + 5) * 3
    adjust_v = ((adjust_v > 255) * 255 + (adjust_v <= 255) * adjust_v).astype("uint8")
    img_hsv[:, :, 2] = adjust_v
    img_soft = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    img_soft = cv2.GaussianBlur(img_soft, (51, 51), 0)
    # Convert to grayscale and create outline sketch effect
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    invert = cv2.bitwise_not(img_gray)
    blur = cv2.GaussianBlur(invert, (21, 21), 0)
    inverted_blur = cv2.bitwise_not(blur)
    sketch = cv2.divide(img_gray, inverted_blur, scale=265.0)
    sketch = cv2.merge([sketch, sketch, sketch])
    # Combine effects to produce watercolor-like result
    img_water = ((sketch / 255.0) * img_soft).astype("uint8")
    cv2.imwrite(output_path, img_water)
    return output_path

def convert_image_to_cartoon(image_path: str, output_path="converted_cartoon.png", num_bilateral=5, sketch_mode=False) -> str:
    # Read the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Downsample image using Gaussian pyramid 
    img_color = img_rgb
    for _ in range(2):
        img_color = cv2.pyrDown(img_color)
    # Apply small bilateral filters repeatedly to remove noise while preserving edges
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
    # Upsample back to original size 
    for _ in range(2):
        img_color = cv2.pyrUp(img_color)
    img_color = cv2.resize(img_color, (img.shape[1], img.shape[0]))
    # Convert to  grayscale and apply median blur
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    # Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(
        img_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=7
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.dilate(edges, kernel, iterations=1)
    # Apply color quantization
    def color_quantization(img, k=8):
        data = np.float32(img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        result = centers[labels.flatten()]
        result = result.reshape(img.shape)
        return result
    img_quant = color_quantization(img_color, k=8)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    if sketch_mode:
        cartoon = edges
    else:
        cartoon = cv2.bitwise_and(img_quant, edges_colored)
    # Convert back to BGR
    if len(cartoon.shape) == 3:
        cartoon_bgr = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
    else:
        cartoon_bgr = cartoon
    cv2.imwrite(output_path, cartoon_bgr)
    return output_path
    
def convert_image_to_glitch(image_path: str, output_path="converted_glitch.png") -> str:
    # Create an ImageGlitcher object
    glitcher = ImageGlitcher()
    # Generate a random seed value
    seed_value = random.randint(1, 20000000)
    random.seed(seed_value)  
    # Load the image
    img = Image.open(image_path).convert("RGB")
    # Apply glitch effect with a random seed and color offset
    glitch_img = glitcher.glitch_image(img, 5, color_offset=True, seed=seed_value)
    # Save the glitched image
    glitch_img.save(output_path)
    return output_path

def convert_image_to_neon_glow(image_path: str, output_path: str = "converted_neon_glow.png") -> str:
    # Open the image and apply the neon glow effect
    with Image.open(image_path).convert("RGB") as img:
        # Find edges in the image
        edges = img.filter(ImageFilter.FIND_EDGES).convert("L")
        edges = ImageEnhance.Brightness(edges).enhance(2.0)
        edges = edges.filter(ImageFilter.GaussianBlur(2))
        edges = ImageOps.equalize(edges)
        # Create a colored version of the edges
        colored_edges = Image.new("RGB", edges.size)
        # Convert to HSV and apply a color gradient
        px_edges = edges.load()
        px_colored = colored_edges.load()
        width, height = edges.size
        # Define the hue range for the gradient
        hue_start = 0.0
        hue_end = 288.0
        hue_range = hue_end - hue_start
        # Convert edges to colored pixels
        for y in range(height):
            for x in range(width):
                intensity = px_edges[x, y] / 255.0
                hue = hue_start + (intensity * hue_range)
                hue_normalized = hue / 360.0
                r, g, b = colorsys.hsv_to_rgb(hue_normalized, 1.0, 1.0)
                px_colored[x, y] = (
                    int(r * 255),
                    int(g * 255),
                    int(b * 255),
                )
        # Apply the neon glow effect and save the output
        neon_glow = ImageChops.screen(img, colored_edges)
        neon_glow.save(output_path)
        return output_path


def convert_image_to_pop_art(image_path: str, output_path: str = "converted_pop_art.jpg"):
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_h, original_w = original_image.shape
    #shrink the image to a smaller size
    max_dots = 100  
    if original_h > original_w:
        ratio = max_dots / float(original_h)
        new_h = max_dots
        new_w = int(original_w * ratio)
    # If the image is wider than it is tall
    else:
        ratio = max_dots / float(original_w)
        new_w = max_dots
        new_h = int(original_h * ratio)
    downsized = cv2.resize(original_image, (new_w, new_h))
    # Reduce multiplier from 100 to 50
    multiplier = 50
    # Prepare final canvas
    canvas_h = new_h * multiplier
    canvas_w = new_w * multiplier
    blank_image = np.full((canvas_h, canvas_w, 3), (102, 0, 102), dtype=np.uint8)
    # For drawing the dots
    padding = multiplier // 2
    dots_colour = (0, 255, 255)
    for y in range(new_h):
        for x in range(new_w):
            intensity = downsized[y, x]
            radius = int((0.7 * multiplier) * ((255 - intensity) / 255.0))
            if radius > 0:
                center = (x * multiplier + padding, y * multiplier + padding)
                cv2.circle(blank_image, center, radius, dots_colour, -1)
    # Resize the image if it's too large for Discord
    MAX_DISCORD_DIM = 1200
    final_h, final_w = blank_image.shape[:2]
    if max(final_w, final_h) > MAX_DISCORD_DIM:
        if final_h > final_w:
            scale = MAX_DISCORD_DIM / float(final_h)
        else:
            scale = MAX_DISCORD_DIM / float(final_w)
        new_w  = int(final_w  * scale)
        new_h  = int(final_h * scale)
        blank_image = cv2.resize(blank_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, blank_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return output_path


def convert_image_to_mosaic(image_path: str, output_path="converted_mosaic.png", block_size=10) -> str:
    img = Image.open(image_path).convert("RGB")
    # Calculate the new width and height
    new_width = max(1, img.width // block_size)
    new_height = max(1, img.height // block_size)
    # Resize the image to a small size
    small_img = img.resize((new_width, new_height), Image.NEAREST)
    mosaic = small_img.resize(img.size, Image.NEAREST)
    mosaic.save(output_path)
    return output_path

def convert_image_to_sepia(image_path: str, output_path: str = "converted_sepia.png") -> str:
    sepia_matrix = (
        0.393, 0.769, 0.189, 0,
        0.349, 0.686, 0.168, 0,
        0.272, 0.534, 0.131, 0
    )
    with Image.open(image_path) as img:
        # Convert to RGB 
        img = img.convert("RGB")
        # Apply the sepia matrix
        sepia_img = img.convert("RGB", sepia_matrix)
        # Save the output
        sepia_img.save(output_path)
    return output_path



