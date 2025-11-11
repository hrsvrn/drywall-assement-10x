import albumentations as A

def get_train_transforms():
    return A.Compose([
        A.CLAHE(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.MotionBlur(p=0.3),
    ])
