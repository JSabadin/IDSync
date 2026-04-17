import argparse
import os
from pathlib import Path
from contextlib import nullcontext

import torch
from PIL import Image
from torchvision import transforms
import onnxruntime as ort

from .clip_utils import CLIPTextModelWrapper, project_face_embs
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer
from .id_pipeline import (
    detect_faces,
    align_and_crop_faces,
    load_retinaface,
    load_model,
)


def add_infer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--arcface_onnx",
        required=True,
        help="Path to the ArcFace ONNX model (.onnx)",
    )
    parser.add_argument(
        "--finetuned_model",
        required=True,
        help="Path to the fine-tuned Stable Diffusion model folder",
    )
    parser.add_argument(
        "--retinaface_lib",
        required=True,
        help="Path to the RetinaFace library folder",
    )
    parser.add_argument(
        "--retinaface_weights",
        required=True,
        help="Path to the RetinaFace weights (.pth)",
    )
    parser.add_argument(
        "--input_image",
        required=True,
        help="Single source image to extract identity from",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where generated images will be saved",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=4,
        help="How many deepfakes to generate",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="Number of DDIM sampling steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (int), or leave unset for nondeterministic",
    )

def parse_args(cli_args=None):
    parser = argparse.ArgumentParser(description="Run ID-Sync inference to generate identity-aware deepfakes")
    add_infer_args(parser)
    return parser.parse_args(cli_args)

def infer_sd(args):
    # fixed settings
    resolution = 224
    base_model = "stabilityai/stable-diffusion-2-1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # prepare output
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # load face detector
    cfg_re50, RetinaFace, decode, decode_landm, PriorBox = load_retinaface(
        args.retinaface_lib
    )
    face_detector = RetinaFace(cfg=cfg_re50, phase='test')
    face_detector = load_model(face_detector, args.retinaface_weights, False)
    face_detector.eval().to(device)

    # load ArcFace ONNX
    session = ort.InferenceSession(args.arcface_onnx)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # load Stable Diffusion components from finetuned checkpoint
    text_encoder = CLIPTextModelWrapper.from_pretrained(
        args.finetuned_model, subfolder="text_encoder", torch_dtype=dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.finetuned_model, subfolder="tokenizer"
    )
    vae = AutoencoderKL.from_pretrained(
        args.finetuned_model, subfolder="vae", torch_dtype=dtype
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.finetuned_model, subfolder="unet", torch_dtype=dtype
    )

    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
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

    # preprocess one image
    img = Image.open(args.input_image).convert("RGB")
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    img_t = t(img).unsqueeze(0).to(device)

    # face detection & alignment
    dets = detect_faces(img_t, face_detector, cfg_re50, decode, decode_landm, PriorBox)
    aligned = align_and_crop_faces(img_t, dets).cpu().numpy()

    # get identity embedding
    out = session.run([output_name], {input_name: aligned})
    id_emb = torch.tensor(out[0], dtype=torch.float32, device=device)
    id_emb = id_emb / id_emb.norm(dim=1, keepdim=True)
    id_emb = project_face_embs(pipeline, id_emb)

    # set up RNG
    generator = (
        torch.Generator(device=device).manual_seed(args.seed)
        if args.seed is not None
        else None
    )

    # autocast
    autocast_ctx = nullcontext() if torch.backends.mps.is_available() else torch.autocast(device)

    # inference loop
    with autocast_ctx:
        for i in range(args.num_images):
            img_out = pipeline(
                prompt_embeds=id_emb,
                height=resolution,
                width=resolution,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]

            fout = out_path / f"deepfake_{i:03d}.jpg"
            img_out.save(fout)
            print(f"[{i+1}/{args.num_images}] saved to {fout}")

    torch.cuda.empty_cache()
