"""
Image -> tensor transforms, shared by training and evaluation.

Keeping both here means preprocessing at training and at evaluation (or any
later inference) cannot drift apart.
"""

from torchvision import transforms

# Statistics of the ImageNet images the pretrained backbones were trained on
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(image_size: int) -> transforms.Compose:
    """Random crop + flip (augmentation), then normalise."""

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    """Resize + center crop (no randomness), then normalise."""

    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
