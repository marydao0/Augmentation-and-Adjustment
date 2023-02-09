from torchvision import transforms as vision_transforms
from torchtext import transforms as text_transforms
import albumentations as A


# pytorch simple augmentation and transformations
img_trans = vision_transforms.Compose([
    vision_transforms.Resize(256),
    vision_transforms.RandomCrop(224),
    vision_transforms.RandomHorizontalFlip(),
    vision_transforms.ToTensor(),
    vision_transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

# https://github.com/albumentations-team/albumentations
# pip install -U albumentations
# albumentations augmentations and transformations

# Composition of more complex data augmentations using the package cited above
aug_img_trans = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.OneOf([
        A.HorizontalFlip(p=1),
        A.RandomRotate90(p=1),
        A.VerticalFlip(p=1)
    ], p=1),
    A.OneOf([
        A.Transpose(p=1),
        A.ShiftScaleRotate(p=1)
    ], p=1),
    A.OneOf([
        A.GridDistortion(p=1),
        A.ElasticTransform(p=1)
    ], p=1),
    A.OneOf([
        A.MotionBlur(p=1),
        A.OpticalDistortion(p=1),
        A.GaussNoise(p=1),
        A.MultiplicativeNoise(p=1)
    ], p=1),
    A.OneOf([
        A.RandomBrightnessContrast(p=1),
        A.RandomGamma(p=1),
        A.RandomGridShuffle(p=1),
        A.RandomRain(p=1),
        A.RandomSunFlare(p=1)
    ], p=1),
    A.ToFloat(),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

aug_img_trans2 = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.OneOf([
        A.HorizontalFlip(p=1),
        A.RandomRotate90(p=1),
        A.VerticalFlip(p=1)
    ], p=1),
    A.OneOf([
        A.Transpose(p=1),
        A.ShiftScaleRotate(p=1)
    ], p=1),
    A.OneOf([
        A.MotionBlur(p=1),
        A.OpticalDistortion(p=1),
        A.GaussNoise(p=1),
        A.MultiplicativeNoise(p=1)
    ], p=1),
    A.ToFloat(),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

# Pytorch text transformation
text_trans = text_transforms.ToTensor().float()
