
import os
import sys
import torch
from torch.nn import functional as F
from scene import Scene, GaussianModel
import random 
from random import randint
from tqdm import tqdm
from gaussian_renderer import render, network_gui
from criteria.clip_loss import CLIPLoss
from criteria.constrative_loss import ContrastiveLoss
from criteria.patchens_loss import PatchNCELoss
from criteria.perp_loss import VGGPerceptualLoss
from utils import io_util
import concurrent.futures
from argparse import ArgumentParser
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams

def create_fine_neg_texts(args):
    path = "criteria/neg_text.txt"
    results = {}
    curr_key = 0
    with open(path, 'r') as fr:
        contents = fr.readlines()
        for item in contents:
            item = item.strip()
            if item.startswith("#"):
                curr_key = item[1:]
                results[curr_key] = []
            else:
                results[curr_key].append(item.split(".")[1])
        
    all_texts = []
    remove_ids = [] 
    ttext = args.finetune.target_text.lower()
    if 'botero' in ttext or 'monalisa' in ttext or 'portrait' in ttext or 'painting' in ttext:
        remove_ids = ['portrait']
    elif 'zombie' in ttext:
        remove_ids = ['zombie']
    elif 'wolf' in ttext:
        remove_ids = ['wolf']
    elif 'pixlar' in ttext or 'disney' in ttext:
        remove_ids = ['disney']
    elif 'sketch' in ttext:
        remove_ids = ['sketch'] 

    for key in results:
        if key not in remove_ids:
        #if key in remove_ids:
            all_texts += results[key]
    return all_texts

def compute_dir_clip_loss(args, loss_dict, rgb_gt, rgb_pred, s_text, t_text):
    dir_clip_loss = loss_dict["clip"](rgb_gt, s_text, rgb_pred, t_text)
    return dir_clip_loss * args.finetune.w_clip

def compute_perceptual_loss(args, loss_dict, rgb_gt, rgb_pred):
    perp_loss = loss_dict["perceptual"](rgb_pred, rgb_gt)
    return perp_loss * args.finetune.w_perceptual

def compute_contrastive_loss(args, loss_dict, rgb_gt, rgb_pred, neg_texts, t_text):
    s_text = random.choice(neg_texts)
    contrastive_loss = loss_dict["contrastive"](rgb_gt, s_text, rgb_pred, t_text)
    return contrastive_loss * args.finetune.w_contrastive

def compute_patch_loss(args, loss_dict, rgb_pred, t_text, neg_texts):
    neg_counts = 8
    s_text_list = random.sample(neg_texts, neg_counts)
    is_full_res = args.data.downscale == 1
    patch_loss = loss_dict["patchnce"](s_text_list, rgb_pred, t_text, is_full_res)
    return patch_loss * args.finetune.w_patchnce

def calc_style_loss(rgb: torch.Tensor, rgb_gt: torch.Tensor, args, loss_dict, neg_texts, H=480):
    """
    Calculate CLIP-driven style losses for Gaussian Splatting.

    Parameters
    ----------
    rgb: torch.Tensor
        Rendered Gaussian splatted image.
    rgb_gt: torch.Tensor
        Ground truth target image.
    args: Namespace
        Argument configuration containing fine-tuning parameters.
    loss_dict: dict
        Dictionary containing different loss functions such as "clip", "perceptual", "contrastive", and "patchnce".
    neg_texts: list
        List of negative sample texts for contrastive loss.
    H: int
        Height of the image, used to reshape the input tensors.
    """
    loss = 0.0
    losses = {"clip": 0.0, "perceptual": 0.0, "contrastive": 0.0, "patchnce": 0.0}

    rgb_pred = rgb.view(-1, *rgb.shape[-3:])
    rgb_gt = rgb_gt.view(-1, *rgb_gt.shape[-3:])

    rgb_pred = F.interpolate(rgb_pred, size=(H, H), mode='bicubic', align_corners=False)
    rgb_gt = F.interpolate(rgb_gt, size=(H, H), mode='bicubic', align_corners=False)
    
    s_text = args.finetune.src_text
    t_text = args.finetune.target_text

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_dir_clip = executor.submit(compute_dir_clip_loss, args, loss_dict, rgb_gt, rgb_pred, s_text, t_text)
        future_perp = executor.submit(compute_perceptual_loss, args, loss_dict, rgb_gt, rgb_pred)
        future_contrastive = executor.submit(compute_contrastive_loss, args, loss_dict, rgb_gt, rgb_pred, neg_texts, t_text)
        future_patch = executor.submit(compute_patch_loss, args, loss_dict, rgb_pred, t_text, neg_texts)
        
        concurrent.futures.wait([future_dir_clip, future_perp, future_contrastive, future_patch], return_when=concurrent.futures.ALL_COMPLETED)

        losses["clip"] = future_dir_clip.result()
        losses["perceptual"] = future_perp.result()
        losses["contrastive"] = future_contrastive.result()
        losses["patchnce"] = future_patch.result()
    
    loss = sum(losses.values()) * args.finetune.w_style

    return loss, losses

def style_training(dataset, opt, pipe, saving_iterations, checkpoint_iterations, checkpoint, config):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    assert checkpoint, "Gaussian model must be exist."

    (model_params, _) = torch.load(checkpoint)
    gaussians.restore(model_params, opt)
    os.makedirs(os.path.join(scene.model_path, "style"), exist_ok=True)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None

    contrastive_loss = ContrastiveLoss()
    patchnce_loss = PatchNCELoss([config.data.reshape_size, config.data.reshape_size]).cuda()
    clip_loss = CLIPLoss()
    perp_loss = VGGPerceptualLoss().cuda()
    loss_dict = {'contrastive': contrastive_loss, 'patchnce': patchnce_loss,\
                 'clip': clip_loss, 'perceptual': perp_loss}
    neg_list = create_fine_neg_texts(config)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()

        style_loss, losses = calc_style_loss(image, gt_image, config, loss_dict, neg_list, H=config.data.reshape_size)
        
        # loss
        style_loss.backward()

        iter_end.record()

        with torch.no_grad():
            if iteration % 10 == 0:
                loss = {
                    "clip": f"{losses['clip']:.{5}f}",
                    "perceptual": f"{losses['perceptual']:.{5}f}",
                    "contrastive": f"{losses['contrastive']:.{5}f}",
                    "patchnce": f"{losses['patchnce']:.{5}f}"
                }
                progress_bar.set_postfix(loss)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/style/chkpnt" + str(iteration) + ".pth")

    print("\n[ITER {}] Saving Checkpoint".format(opt.iterations))
    torch.save((gaussians.capture(), opt.iterations), scene.model_path + "/style/chkpnt" + str(opt.iterations) + ".pth")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 14_000, 21_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--config", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    config = io_util.load_config(args)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    style_training(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, config)

    # All done
    print("\nTraining complete.")