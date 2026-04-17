import argparse
import os
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import contextlib

from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from transformers import CLIPTokenizer
from .clip_utils import CLIPTextModelWrapper, project_face_embs
from .id_pipeline import (
    detect_faces,
    align_and_crop_faces,
    load_retinaface,
    load_model,
)


def add_synth_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--pca_dir",
        required=True,
        help="Directory containing pca_mean.npy and pca_components.npy",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Where to save the synthetic dataset",
    )
    parser.add_argument(
        "--finetuned_model_path",
        required=True,
        help="Path to the fine-tuned Stable Diffusion model folder",
    )
    parser.add_argument(
        "--face_detector_weights",
        required=True,
        help="Path to the RetinaFace weights (.pth)",
    )
    parser.add_argument(
        "--pytorch_retinaface_library_path",
        required=True,
        help="Path to the RetinaFace library folder",
    )
    parser.add_argument(
        "--seed", type=int, default=1441, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=25, help="Diffusion steps"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=3.0, help="Guidance scale"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Generation batch size"
    )
    parser.add_argument(
        "--num_ids", type=int, default=10000, help="Number of identities"
    )
    parser.add_argument(
        "--images_per_id", type=int, default=50, help="Images per identity"
    )
    parser.add_argument(
        "--noise_variance",
        type=float,
        default=0.001,
        help="Intra-class embedding noise variance σ² added to each identity center"
    )
    parser.add_argument(
        "--resolution", type=int, default=224, help="Output image resolution"
    )


def parse_args(cli_args=None):
    parser = argparse.ArgumentParser(
        description="Create synthetic dataset for ID-Sync"
    )
    add_synth_args(parser)
    return parser.parse_args(cli_args)


def create_synth_dataset(args):
    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    # load PCA
    pca_mean = np.load(os.path.join(args.pca_dir, "pca_mean.npy"))
    pca_components = np.load(os.path.join(args.pca_dir, "pca_components.npy"))
    pca_variance = np.load(os.path.join(args.pca_dir, "pca_explained_variance.npy"))

    def sample_pca_embeddings(num_samples):
        n_components = pca_components.shape[0]

        z0 = np.random.normal(0, 1, size=(num_samples, n_components))
        z = z0 * np.sqrt(pca_variance)[None, :]      # shape (num_samples, k)
        emb = z.dot(pca_components) + pca_mean      # shape (num_samples, dim)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return torch.tensor(emb, dtype=torch.float16).to(device)


    # prepare output
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    model_name = Path(args.finetuned_model_path).name

    # load face detector
    cfg_re50, RetinaFace, decode, decode_landm, PriorBox = load_retinaface(
        args.pytorch_retinaface_library_path
    )
    face_detector = RetinaFace(cfg=cfg_re50, phase="test")
    face_detector = load_model(
        face_detector, args.face_detector_weights, False
    ).eval().to(device)

    # load Stable Diffusion
    text_encoder = CLIPTextModelWrapper.from_pretrained(
        args.finetuned_model_path, subfolder="text_encoder", torch_dtype=dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.finetuned_model_path, subfolder="tokenizer"
    )
    vae = AutoencoderKL.from_pretrained(
        args.finetuned_model_path, subfolder="vae", torch_dtype=dtype
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.finetuned_model_path, subfolder="unet", torch_dtype=dtype
    )

    pipeline = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=unet,
        safety_checker=None,
        torch_dtype=dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    pipeline = pipeline.to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    total = args.num_ids * args.images_per_id

    with tqdm(total=total, desc="Generating Images") as pbar:
        for id_num in range(args.num_ids):
            align_dir = os.path.join(output_dir, f"{model_name}_align", str(id_num))
            no_align_dir = os.path.join(
                output_dir, f"{model_name}_no_align", str(id_num)
            )
            os.makedirs(align_dir, exist_ok=True)
            os.makedirs(no_align_dir, exist_ok=True)

            id_center = sample_pca_embeddings(1)

            gen_count = 0
            while gen_count < args.images_per_id:
                bs = min(args.batch_size, args.images_per_id - gen_count)
                eps = torch.randn((bs, id_center.shape[1]), device=device, dtype=id_center.dtype) * (args.noise_variance ** 0.5)
                id_emb = id_center.repeat(bs, 1) + eps
                id_emb = id_emb / id_emb.norm(dim=1, keepdim=True)
                id_emb = project_face_embs(pipeline, id_emb)

                with contextlib.redirect_stdout(open(os.devnull, "w")), contextlib.redirect_stderr(open(os.devnull, "w")):
                    imgs = pipeline(
                        prompt_embeds=id_emb,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=generator,
                        num_images_per_prompt=bs,
                        height=args.resolution,
                        width=args.resolution,
                    ).images

                for img in imgs:
                    if gen_count >= args.images_per_id:
                        break
                    img.save(os.path.join(no_align_dir, f"{gen_count}.jpg"))
                    img_t = transforms.ToTensor()(img).unsqueeze(0).to(device)
                    det = detect_faces(img_t, face_detector, cfg_re50, decode, decode_landm, PriorBox)
                    if det is not None:
                        aligned = align_and_crop_faces(img_t, det)
                        out_img = transforms.ToPILImage()(aligned.squeeze().cpu())
                    else:
                        out_img = img
                    out_img.save(os.path.join(align_dir, f"{gen_count}.jpg"))
                    gen_count += 1
                    pbar.update(1)

    print("Synthetic dataset created at", output_dir)


if __name__ == "__main__":
    args = parse_args()
    create_synth_dataset(args)
