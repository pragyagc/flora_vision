import os
import cv2
import albumentations as A


# PATHS

input_dir = "D:/data"
output_dir = "E:/FINAL_PROJECT/augmented"
os.makedirs(output_dir, exist_ok=True)

IMG_SIZE = 224
AUG_PER_IMAGE = 4  


# AUGMENTATION PIPELINE

augment = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),

    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),

    A.Rotate(limit=20, p=0.5),

    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=0,
        p=0.5
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),

    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=15,
        val_shift_limit=10,
        p=0.4
    ),

    A.GaussianBlur(blur_limit=3, p=0.2),
])


# PROCESS DATASET

for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    save_class_dir = os.path.join(output_dir, class_name)
    os.makedirs(save_class_dir, exist_ok=True)

    for filename in os.listdir(class_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(class_path, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping corrupted image: {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        base_name = os.path.splitext(filename)[0]

        # Generate augmented images
        for i in range(AUG_PER_IMAGE):
            augmented = augment(image=img)
            aug_img = augmented["image"]

            save_path = os.path.join(
                save_class_dir,
                f"{base_name}_{i}.png"
            )

            # Save lossless
            cv2.imwrite(
                save_path,
                cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_PNG_COMPRESSION, 0]
            )

print(" CNN preprocessing + augmentation completed!")
