
import os
import sys
import uuid
import torch
import torch.nn.functional as F
from CLIPStyler import StyleTransfer
from scene import Scene, GaussianModel
from random import randint
from tqdm import tqdm
from gaussian_renderer import render, network_gui
from criteria.clip_loss import CLIPLoss
from criteria.constrative_loss import ContrastiveLoss
from utils import io_util
import concurrent.futures
from argparse import ArgumentParser, Namespace
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

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
    # remove_ids = [] 
    # ttext = args.finetune.target_text.lower()
    # if 'botero' in ttext or 'monalisa' in ttext or 'portrait' in ttext or 'painting' in ttext:
    #     remove_ids = ['portrait']
    # elif 'zombie' in ttext:
    #     remove_ids = ['zombie']
    # elif 'wolf' in ttext:
    #     remove_ids = ['wolf']
    # elif 'pixlar' in ttext or 'disney' in ttext:
    #     remove_ids = ['disney']
    # elif 'sketch' in ttext:
    #     remove_ids = ['sketch'] 

    # for key in results:
    #     if key not in remove_ids:
    #     #if key in remove_ids:
    #         all_texts += results[key]

    all_texts = results["base"]
    return all_texts

def compute_dir_clip_loss(args, loss_dict, rgb_gt, rgb_pred, s_text, t_text, mask, stylized):
    dir_clip_loss = loss_dict["clip"](rgb_gt, s_text, rgb_pred.clone() * mask, t_text)
    Ll1 = l1_loss(F.interpolate(rgb_pred, (512, 512), mode="bicubic", align_corners=False), stylized)
    return (dir_clip_loss * 0.3 + Ll1 * 0.7) * args.finetune.w_clip * mask.float().mean()

def compute_perceptual_loss(args, rgb_gt, rgb_pred, opt):
    loss = opt.lambda_dssim * (1.0 - ssim(rgb_pred, rgb_gt))
    return loss * args.finetune.w_perceptual

def compute_smoothness_loss(args, rgb_pred, mask):
    rgb_pred = rgb_pred.clone() * mask
    laplacian_kernel = torch.tensor([[[[0, 1, 0], 
                                        [1, -4, 1], 
                                        [0, 1, 0]]]], dtype=torch.float32, device=rgb_pred.device)
    laplacian_kernel = laplacian_kernel.repeat(*rgb_pred.shape[:2], 1, 1)
    laplacian_output = F.conv2d(rgb_pred, laplacian_kernel, padding=1, groups=1)
    
    laplacian_loss = torch.abs(laplacian_output).mean()
    return laplacian_loss * args.finetune.w_smooth * mask.float().mean()

def compute_sharpness_loss(args, rgb_pred, rgb_gt, mask):
    def rgb_to_grayscale(image):
        return 0.2989 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.1140 * image[:, 2, :, :]

    rgb_pred = rgb_to_grayscale(rgb_pred.clone() * mask).unsqueeze(1)
    rgb_gt = rgb_to_grayscale(rgb_gt).unsqueeze(1)

    sobel_x = torch.tensor([[[[-1, 0, 1], 
                              [-2, 0, 2], 
                              [-1, 0, 1]]]], dtype=torch.float32, device=rgb_pred.device)
    sobel_y = torch.tensor([[[[-1, -2, -1], 
                              [ 0,  0,  0], 
                              [ 1,  2,  1]]]], dtype=torch.float32, device=rgb_pred.device)

    sobel_x = sobel_x.repeat(*rgb_pred.shape[:2], 1, 1)
    sobel_y = sobel_y.repeat(*rgb_pred.shape[:2], 1, 1)

    grad_pred_x = F.conv2d(rgb_pred, sobel_x, padding=1, groups=1)
    grad_pred_y = F.conv2d(rgb_pred, sobel_y, padding=1, groups=1)
    grad_gt_x = F.conv2d(rgb_gt, sobel_x, padding=1, groups=1)
    grad_gt_y = F.conv2d(rgb_gt, sobel_y, padding=1, groups=1)
    
    grad_diff_x = torch.abs(grad_pred_x - grad_gt_x)
    grad_diff_y = torch.abs(grad_pred_y - grad_gt_y)
    
    sharpness_loss = (grad_diff_x + grad_diff_y).mean()
    return sharpness_loss * args.finetune.w_sharp * mask.float().mean()

def find_nth_occurrence(lst, n):
    count = -1
    for i, v in enumerate(lst):
        if v:
            count += 1
            if count == n:
                lst[i] = False
                return i
    return -1 

def calc_style_loss(rgb: torch.Tensor, rgb_gt: torch.Tensor, args, loss_dict, stylized, opt, background):
    """
    Calculate CLIP-driven style losses for Gaussian Splatting.
    """
    loss = 0.0
    losses = {"clip": 0.0, "perceptual": 0.0, "contrastive": 0.0, "smooth": 0.0, "sharp": 0.0}

    rgb_pred = rgb.view(-1, *rgb.shape[-3:])
    rgb_gt = rgb_gt.view(-1, *rgb_gt.shape[-3:])
    
    s_text = args.finetune.src_text
    t_text = args.finetune.target_text

    background = background.view(1, 3, 1, 1)
    mask = (rgb_gt != background).any(dim=1, keepdim=True).repeat(1, 3, 1, 1)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_dir_clip = executor.submit(compute_dir_clip_loss, args, loss_dict, rgb_gt, rgb_pred, s_text, t_text, mask, stylized)
        future_perp = executor.submit(compute_perceptual_loss, args, rgb_gt, rgb_pred, opt)
        future_smoothness = executor.submit(compute_smoothness_loss, args, rgb_pred, mask)
        future_sharpness = executor.submit(compute_sharpness_loss, args, rgb_pred, rgb_gt, mask)

        concurrent.futures.wait([future_dir_clip, future_perp, future_smoothness, future_sharpness], return_when=concurrent.futures.ALL_COMPLETED)

        losses["clip"] = future_dir_clip.result()
        losses["perceptual"] = future_perp.result()
        losses["smooth"] = future_smoothness.result()
        losses["sharp"] = future_sharpness.result()
    
    loss = sum(losses.values()) * args.finetune.w_style

    return loss, losses

def style_training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, config):
    style_net = StyleTransfer(os.path.join(dataset.source_path, dataset.images), config.finetune.target_text, config.finetune.src_text, 12)
    style_net.train()

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, style_train=True)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    assert checkpoint, "Gaussian model must be exist."
    (model_params, _) = torch.load(checkpoint)
    gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    index_stack = None
    stylized_stack = {}
    ema_loss_for_log = 0.0

    # contrastive_loss = ContrastiveLoss(distance_type="cosine")
    # clip_loss = CLIPLoss()
    # loss_dict = {'contrastive': contrastive_loss, 'clip': clip_loss}
    # neg_list = create_fine_neg_texts(config)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            index_stack = [True for _ in range(len(viewpoint_stack))]
        
        idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(idx)
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        idx = find_nth_occurrence(index_stack, idx)
        if stylized_stack.get(idx) is None:
            stylized_stack[idx] = style_net.stylize(gt_image.view(-1, *gt_image.shape[-3:]))

        Ll1 = l1_loss(image, stylized_stack[idx])
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = opt.lambda_normal * (normal_error).mean()
        dist_loss = opt.lambda_dist * (rend_dist).mean()

        # style_loss, losses = calc_style_loss(image, gt_image, config, loss_dict, stylized_stack[idx], opt, background)

        loss = loss + normal_loss + dist_loss
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                l_dict = {
                    "loss": f"{loss:.{5}f}",
                    "normal_loss": f"{normal_loss:.{5}f}",
                    "dist_loss": f"{dist_loss:.{5}f}"
                }
                progress_bar.set_postfix(l_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
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
    style_training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, config)

    # All done
    print("\nTraining complete.")