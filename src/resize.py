from PIL import Image
import os

def resize_images(input_dir, output_dir, size=(600, 600)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_name in os.listdir(input_dir):
        if image_name.endswith(".jpg") or image_name.endswith(".png"):
            image_path = os.path.join(input_dir, image_name)
            try:
                with Image.open(image_path) as img:
                    img = img.resize(size)
                    optimized_image_path = os.path.join(output_dir, image_name)
                    img.save(optimized_image_path, optimize=True, quality=85)
                    print(f"Resized and optimized {image_name}")
            except Exception as e:
                print(f"Error resizing {image_name}: {e}")

def main():
    input_dir = "./images"
    output_dir = "./resized_images"
    resize_images(input_dir, output_dir)
    
if __name__ == '__main__':
    main()