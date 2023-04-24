import os
import re
from os import listdir
from os.path import isfile, join

import torch
import torchvision.models as models
from PIL import Image
from torchvision.models import VGG19_Weights
from torchvision.utils import save_image

from utils import get_num, image_loader, run_style_transfer

# Generate a folder to save results
if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device("cuda")
    torch.cuda.amp.GradScaler()
elif torch.backends.mps.is_available():
    print("MPS is available")
    # device = torch.device("mps")
    device = torch.device("cpu")
else:
    print("CPU is available")
    device = torch.device("cpu")

# read all images that contain the word "style" in the folder
STYLE_PATH = "../Results/Task_e/mask_RCNN/"
style_images = [f for f in listdir(STYLE_PATH) if isfile(join(STYLE_PATH, f)) and "style" in f]
# sorted list
style_images = sorted(style_images, key=get_num)
# read all images that contain the word "pred" in the folder
PRED_PATH = "../Results/Task_e/mask_RCNN/"
content_images = [f for f in listdir(PRED_PATH) if isfile(join(PRED_PATH, f)) and "pred" in f]
# sorted list
content_images = sorted(content_images, key=get_num)

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

NUM_STEPS = 600
CONTENT_WEIGHT = 1
STYLE_WEIGHT = 100000
CURRENT_PATH = os.getcwd()
RESULT_PATH = os.path.join(CURRENT_PATH, "../Results/Task_e/style_transfer")
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)

cnn = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()

for ii, (style, content) in enumerate(zip(style_images, content_images)):
    # relevant path
    style = join(STYLE_PATH, style)
    content = join(PRED_PATH, content)

    # print(style)
    # print(content)

    aux = Image.open(content)
    content_size = aux.size
    print(content_size)

    if min(content_size) > 1000:
        content_size = (800, 800)

    style_img = image_loader(style, min(content_size))
    content_img = image_loader(content, min(content_size))

    assert style_img.size() == content_img.size(), "style & content imgs should be same size"

    # input image as initializer
    input_img = content_img.clone()
    # uncomment & replace 4 white noise initializer

    """
    input_img = torch.randn(content_img.data.size(), device=device, requires_grad=False)
    # Gaussian smooth

    blur = cv2.GaussianBlur(input_img.detach().squeeze().cpu().numpy(), (3, 3), 0)
    blur = blur - blur.min()
    blur = blur / blur.max()
    blur = torch.from_numpy(blur).to(device).unsqueeze(0)
    """

    output = run_style_transfer(
        cnn,
        cnn_normalization_mean,
        cnn_normalization_std,
        content_img,
        style_img,
        input_img,
        num_steps=NUM_STEPS,
        print_step=100,
        content_weight=CONTENT_WEIGHT,
        style_weight=STYLE_WEIGHT,
    )
    # output = output.reshape(content_size)
    # split .jpg from name
    style = re.sub('[^0-9]', '', style)
    content = re.sub('[^0-9]', '', content)
    print("SAVING IMAGE, STYLE: ", style, " CONTENT: ", content)
    save_image(output, os.path.join(RESULT_PATH, f'{style + "_+_" + content}.png'))
    torch.cuda.empty_cache()

print("PROCESS FINISHED")
