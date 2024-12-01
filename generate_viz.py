from PIL import Image, ImageDraw
import numpy as np

def blend_images(image1_path, image2_path, output_path, opacity=0.9):
    # Load the images
    image1 = Image.open(image1_path).convert("RGBA")
    image2 = Image.open(image2_path).convert("RGBA")
    
    # Resize images to have the same dimensions
    image1 = image1.resize(image2.size)
    image2 = image2.resize(image1.size)
    width, height = image1.size

    # Create a blank image for the output
    output = Image.new("RGBA", (width, height))

    # Create masks for the triangular regions
    lower_triangle_mask = Image.new("L", (width, height), 0)
    upper_triangle_mask = Image.new("L", (width, height), 0)
    draw_lower = ImageDraw.Draw(lower_triangle_mask)
    draw_upper = ImageDraw.Draw(upper_triangle_mask)

    # Define triangular regions
    draw_lower.polygon([(0, height), (width, height), (0, 0)], fill=int(255*opacity))  # Lower triangle
    draw_upper.polygon([(0, 0), (width, 0), (width, height)], fill=255)  # Upper triangle

    # Apply the masks to the respective images
    lower_triangle = Image.composite(image1, Image.new("RGBA", (width, height), (0, 0, 0, 0)), lower_triangle_mask)
    upper_triangle = Image.composite(image2, Image.new("RGBA", (width, height), (0, 0, 0, 0)), upper_triangle_mask)

    # Combine the two halves into the output image
    output = Image.alpha_composite(lower_triangle, upper_triangle)

    # Save the result
    output.save(output_path, format="PNG")
    print(f"Blended image saved to {output_path}")

# Example usage
blend_images('/data5/GSLoc-Unofficial-Implementation/tmp/fire_marepo.png', '/data5/GSLoc-Unofficial-Implementation/tmp/fire_gt.png', '/data5/GSLoc-Unofficial-Implementation/tmp/fire_marepo_viz.jpg')