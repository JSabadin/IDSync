import os
import argparse
import numpy as np
import torch
import PIL
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
import onnxruntime as ort
from .clip_utils import CLIPTextModelWrapper, project_face_embs

def parse_args(cli_args=None):
    parser = argparse.ArgumentParser(
        description="Precompute dataset for text-to-image embeddings"
    )
    parser.add_argument(
        "--arcface_onnx",
        type=str,
        required=True,
        help="Path to arcface ONNX model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        help="Stable Diffusion base model name or path"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing images"
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        required=True,
        help="Output directory for the generated prompts"
    )
    return parser.parse_args(cli_args)

def precompute_dataset(opts):
    weight_dtype = torch.float32
    pipeline = StableDiffusionPipeline.from_pretrained(
        opts.base_model,
        torch_dtype=weight_dtype
    )
    wrapped_clip = CLIPTextModelWrapper.from_pretrained(
        opts.base_model,
        subfolder="text_encoder",
        torch_dtype=weight_dtype
    )
    pipeline.text_encoder = wrapped_clip
    pipeline = pipeline.to('cuda')

    session = ort.InferenceSession(opts.arcface_onnx)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    def validate_pt_file(file_path):
        try:
            data = torch.load(file_path)
            return isinstance(data, torch.Tensor)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return False

    def get_prompt_path(image_path):
        relative_path = os.path.relpath(image_path, opts.image_dir)
        prompt_path = os.path.join(opts.prompt_dir, relative_path)
        base, _ = os.path.splitext(prompt_path)
        return base + ".pt"

    print("Gathering all image files...")
    image_files = []
    for root, _, files in os.walk(opts.image_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                image_files.append(os.path.join(root, file))
    print(f"Found {len(image_files)} total image files.")

    print("Validating existing embeddings...")
    remaining_images = []
    existing_count = 0
    for img_path in tqdm(image_files):
        pt_path = get_prompt_path(img_path)
        if os.path.exists(pt_path) and validate_pt_file(pt_path):
            existing_count += 1
        else:
            remaining_images.append(img_path)
    print(f"Found {existing_count} valid existing embeddings.")
    print(f"{len(remaining_images)} images remaining to process.")

    for image_path in tqdm(remaining_images, desc="Processing images"):
        image = PIL.Image.open(image_path).convert("RGB").resize((112, 112), PIL.Image.BILINEAR)
        image = np.array(image)

        aligned_face = image.astype(np.float32) / 255.0
        aligned_face = aligned_face * 2.0 - 1.0
        aligned_face = aligned_face.transpose(2, 0, 1)
        aligned_face = aligned_face[np.newaxis, ...]

        output = session.run([output_name], {input_name: aligned_face})
        id_emb = torch.tensor(output[0], dtype=torch.float32).cuda()
        id_emb = id_emb / torch.norm(id_emb, dim=1, keepdim=True)

        token_embeds = project_face_embs(pipeline, id_emb, only_token_embs=True)

        relative_path = os.path.relpath(image_path, opts.image_dir)
        prompt_path = os.path.join(opts.prompt_dir, relative_path)
        prompt_dirname = os.path.dirname(prompt_path)
        os.makedirs(prompt_dirname, exist_ok=True)

        torch.save(
            token_embeds.cpu(),
            prompt_path.replace(".jpg", ".pt").replace(".png", ".pt").replace(".jpeg", ".pt")
        )

    print("Precomputed prompts saved successfully.")

if __name__ == "__main__":
    precompute_dataset(parse_args())
