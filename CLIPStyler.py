import os
import torch
import torch.nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
import StyleNet
import clip
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
import matplotlib.pyplot as plt

class StyleTransfer:
    def __init__(self, content_path: str, target_text: str, source_text: str, num_crops: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.VGG = models.vgg19(pretrained=True).features.to(self.device)
        for parameter in self.VGG.parameters():
            parameter.requires_grad_(False)

        self.content_path = content_path
        self.target_text = target_text
        self.source_text = source_text
        self.num_crops = num_crops
        self.crop_size = 128
        self.lambda_c = 150
        self.lambda_patch = 9000
        self.lambda_dir = 500
        self.lambda_tv = 2e-3
        self.lambda_rgb = 2000
        self.lr = 5e-4
        self.max_step = 40
        self.thresh = 0.7

        self.style_net = StyleNet.UNet().to(self.device)
        self.optimizer = optim.Adam(self.style_net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.max_step // 3, gamma=0.5)

        self.cropper = transforms.Compose([
            transforms.RandomCrop(self.crop_size)
        ])
        self.augment = transforms.Compose([
            transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
            transforms.Resize(224)
        ])

        self.imagenet_templates = [
            'a bad photo of a {}.',
            'a sculpture of a {}.',
            'a photo of the hard to see {}.',
            'a low resolution photo of the {}.',
            'a rendering of a {}.',
            'graffiti of a {}.',
            'a bad photo of the {}.',
            'a cropped photo of the {}.',
            'a tattoo of a {}.',
            'the embroidered {}.',
            'a photo of a hard to see {}.',
            'a bright photo of a {}.',
            'a photo of a clean {}.',
            'a photo of a dirty {}.',
            'a dark photo of the {}.',
            'a drawing of a {}.',
            'a photo of my {}.',
            'the plastic {}.',
            'a photo of the cool {}.',
            'a close-up photo of a {}.',
            'a black and white photo of the {}.',
            'a painting of the {}.',
            'a painting of a {}.',
            'a pixelated photo of the {}.',
            'a sculpture of the {}.',
            'a bright photo of the {}.',
            'a cropped photo of a {}.',
            'a plastic {}.',
            'a photo of the dirty {}.',
            'a jpeg corrupted photo of a {}.',
            'a blurry photo of the {}.',
            'a photo of the {}.',
            'a good photo of the {}.',
            'a rendering of the {}.',
            'a {} in a video game.',
            'a photo of one {}.',
            'a doodle of a {}.',
            'a close-up photo of the {}.',
            'a photo of a {}.',
            'the origami {}.',
            'the {} in a video game.',
            'a sketch of a {}.',
            'a doodle of the {}.',
            'a origami {}.',
            'a low resolution photo of a {}.',
            'the toy {}.',
            'a rendition of the {}.',
            'a photo of the clean {}.',
            'a photo of a large {}.',
            'a rendition of a {}.',
            'a photo of a nice {}.',
            'a photo of a weird {}.',
            'a blurry photo of a {}.',
            'a cartoon {}.',
            'art of a {}.',
            'a sketch of the {}.',
            'a embroidered {}.',
            'a pixelated photo of a {}.',
            'itap of the {}.',
            'a jpeg corrupted photo of the {}.',
            'a good photo of a {}.',
            'a plushie {}.',
            'a photo of the nice {}.',
            'a photo of the small {}.',
            'a photo of the weird {}.',
            'the cartoon {}.',
            'art of the {}.',
            'a drawing of the {}.',
            'a photo of the large {}.',
            'a black and white photo of a {}.',
            'the plushie {}.',
            'a dark photo of a {}.',
            'itap of a {}.',
            'graffiti of the {}.',
            'a toy {}.',
            'itap of my {}.',
            'a photo of a cool {}.',
            'a photo of a small {}.',
            'a tattoo of the {}.',
        ]

        self.clip_model, self.preprocess = clip.load('ViT-B/32', self.device, jit=False)

    @staticmethod
    def img_normalize(image):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to(image)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(image)
        return (image - mean) / std

    @staticmethod
    def img_denormalize(image):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to(image)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(image)
        return image * std + mean

    @staticmethod
    def clip_normalize(image):
        image = F.interpolate(image, size=224, mode='bicubic')
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, -1, 1, 1).to(image)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, -1, 1, 1).to(image)
        return (image - mean) / std

    @staticmethod
    def get_image_prior_losses(inputs_jit):
        diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
        diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
        diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
        diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
        return sum(torch.norm(diff) for diff in [diff1, diff2, diff3, diff4])

    def compose_text_with_templates(self, text):
        return [template.format(text) for template in self.imagenet_templates]

    def load_images(self, img_path):
        images = []
        for img in os.listdir(img_path):
            image = Image.open(os.path.join(img_path, img)).resize((512, 512))
            transform = transforms.ToTensor()
            images.append(transform(image)[:3, :, :].unsqueeze(0).to(self.device))
        return images
    
    def get_clip_text_features(self, template_text):
        tokens = clip.tokenize(template_text).to(self.device)
        text_features = self.clip_model.encode_text(tokens).mean(dim=0, keepdim=True)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def calculate_patch_loss(self, target, source_features, text_features, text_source):
        img_proc = torch.cat([self.augment(self.cropper(target)) for _ in range(self.num_crops)], dim=0)
        image_features = self.clip_model.encode_image(self.clip_normalize(img_proc))
        image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        img_direction = image_features - source_features
        img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

        text_direction = (text_features - text_source).repeat(image_features.size(0), 1)
        text_direction /= text_direction.clone().norm(dim=-1, keepdim=True)

        loss = (1 - torch.cosine_similarity(img_direction, text_direction, dim=1))
        loss[loss < self.thresh] = 0
        return loss.mean()

    def calculate_global_loss(self, target, source_features, text_features, text_source):
        glob_features = self.clip_model.encode_image(self.clip_normalize(target))
        glob_features /= glob_features.clone().norm(dim=-1, keepdim=True)

        glob_direction = glob_features - source_features
        glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)

        text_direction = (text_features - text_source)
        return (1 - torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()

    def get_features(self, image, model):
        layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1', '31': 'conv5_2'}
        features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def train(self):
        images = self.load_images(self.content_path)
        prompt = self.target_text
        source = self.source_text
        self.VGG.to(self.device)

        with torch.no_grad():
            template_text = self.compose_text_with_templates(prompt)
            text_features = self.get_clip_text_features(template_text)

            template_source = self.compose_text_with_templates(source)
            text_source = self.get_clip_text_features(template_source)

        for step in tqdm(range(self.max_step + 1)):
            for content_image in images:
                content_features = self.get_features(self.img_normalize(content_image), self.VGG)

                source_features = self.clip_model.encode_image(self.clip_normalize(content_image))
                source_features /= source_features.clone().norm(dim=-1, keepdim=True)

                target = self.style_net(content_image, use_sigmoid=True).to(self.device)
                target.requires_grad_(True)

                target_features = self.get_features(self.img_normalize(target), self.VGG)

                content_loss = sum(torch.mean((target_features[layer] - content_features[layer]) ** 2)
                                  for layer in ['conv4_2', 'conv5_2'])
                loss_patch = self.calculate_patch_loss(target, source_features, text_features, text_source)
                loss_glob = self.calculate_global_loss(target, source_features, text_features, text_source)

                reg_tv = self.lambda_tv * self.get_image_prior_losses(target)

                color_preservation_loss = F.l1_loss(target, content_image.to(self.device))

                total_loss = self.lambda_patch * loss_patch + self.lambda_c * content_loss + reg_tv + self.lambda_dir * loss_glob + self.lambda_rgb * color_preservation_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
            self.scheduler.step()

            if step % 20 == 0:
                output_image = target.clone()
                output_image = torch.clamp(output_image, 0, 1)
                output_image = TF.adjust_contrast(output_image, 1.5).squeeze(0)
                os.makedirs("./StyleNet_OUTPUT", exist_ok=True)
                plt.imshow(TF.to_pil_image(output_image))
                plt.savefig(f"./StyleNet_OUTPUT/{step}.png")
                plt.show()
        
        self.VGG.to("cpu")

    def stylize(self, img: torch.Tensor):
        with torch.no_grad():
            result = self.style_net(F.interpolate(img, (512, 512), mode="bicubic", align_corners=False), use_sigmoid=True).to("cpu")
            result = torch.clamp(result, 0, 1).squeeze(0)
        return TF.to_tensor(TF.to_pil_image(result)).to(img).unsqueeze(0)