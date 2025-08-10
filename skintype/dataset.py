import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

def augment_class_images(class_dir, target_num_images, prefix="aug", batch_size=1):
    img_extensions = ('.png', '.jpg', '.jpeg')
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(img_extensions)]
    current_num_images = len(images)
    if current_num_images == 0:
        print(f"No images in {class_dir}. Skipping.")
        return
    if current_num_images >= target_num_images:
        print(f"Class '{os.path.basename(class_dir)}' has {current_num_images} images, target is {target_num_images}. Skipping.")
        return
    print(f"Augmenting '{os.path.basename(class_dir)}': {current_num_images} -> {target_num_images}")
    datagen = ImageDataGenerator(
        rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
        shear_range=0.1, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
    )
    needed = target_num_images - current_num_images
    generated = 0
    image_index = 0
    while generated < needed:
        img_name = images[image_index % current_num_images]
        image_index += 1
        img_path = os.path.join(class_dir, img_name)
        img = load_img(img_path)
        x = img_to_array(img).reshape((1,) + img_to_array(img).shape)
        aug_iter = datagen.flow(x, batch_size=batch_size, save_to_dir=class_dir,
                                save_prefix=prefix, save_format='jpg')
        for _ in range(batch_size):
            if generated >= needed:
                break
            next(aug_iter)
            generated += 1
    print(f"Finished augmenting '{os.path.basename(class_dir)}' with {needed} new images.")

if __name__ == "__main__":
    train_dir = r'C:\Users\LENOVO\OneDrive\Desktop\beautyintelligent\skintype\dataset\train'
    img_extensions = ('.png', '.jpg', '.jpeg')
    class_counts = {}
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if f.lower().endswith(img_extensions)]
            class_counts[class_name] = len(images)
    print("Current class distribution:")
    for cls, cnt in class_counts.items():
        print(f"  {cls}: {cnt} images")
    max_count = max(class_counts.values())
    print(f"\nTarget images per class for balance: {max_count}\n")
    for cls, count in class_counts.items():
        augment_class_images(os.path.join(train_dir, cls), max_count)
    print("Dataset augmentation complete.")