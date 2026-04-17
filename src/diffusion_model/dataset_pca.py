import argparse
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
import onnxruntime as ort

def parse_args():
    parser = argparse.ArgumentParser(description="Run PCA on ArcFace embeddings.")
    parser.add_argument("--arcface_onnx_path", required=True, help="Path to ArcFace ONNX model.")
    parser.add_argument("--dataset_root", required=True, help="Root directory of the image dataset.")
    parser.add_argument("--output_dir", required=True, help="Directory to save PCA outputs.")
    parser.add_argument("--json_mapping", required=True, help="Path to JSON with folder-to-ID mapping.")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size for incremental PCA.")
    parser.add_argument("--pca_components", type=int, default=400, help="Number of PCA components.")
    return parser.parse_args()

def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((112, 112))
        img = np.array(img).transpose(2, 0, 1)
        img = (img - 127.5) / 128.0
        return img.astype(np.float32)
    except Exception:
        return None

def process_folder(folder_path, session, input_name, batch, pbar):
    for img_file in tqdm(os.listdir(folder_path), desc=f"Processing {os.path.basename(folder_path)}", leave=False):
        img_path = os.path.join(folder_path, img_file)
        preprocessed = preprocess_image(img_path)
        if preprocessed is not None:
            emb = session.run(None, {input_name: preprocessed[np.newaxis, ...]})[0][0]
            batch.append(emb / np.linalg.norm(emb))
            pbar.update(1)
    return batch

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    session = ort.InferenceSession(args.arcface_onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    ipca = IncrementalPCA(n_components=args.pca_components, batch_size=args.batch_size)
    mmap_path = os.path.join(args.output_dir, "temp_embeddings.dat")
    mmap_emb = np.memmap(mmap_path, dtype=np.float32, mode='w+', shape=(args.batch_size, 512))

    with open(args.json_mapping, 'r') as f:
        folder_to_id_mapping = json.load(f)
    valid_folders = set(folder_to_id_mapping.keys())

    total_images = 0
    current_batch = []
    mean = None

    main_pbar = tqdm(desc="Total progress", unit="img")

    for folder_name in tqdm(sorted(os.listdir(args.dataset_root), key=lambda x: int(x)), desc="Processing folders"):
        folder_path = os.path.join(args.dataset_root, folder_name)
        if folder_name not in valid_folders or not os.path.isdir(folder_path):
            continue

        current_batch = process_folder(folder_path, session, input_name, current_batch, main_pbar)

        while len(current_batch) >= args.batch_size:
            batch_array = np.array(current_batch[:args.batch_size])
            mean = batch_array.mean(axis=0) if mean is None else (mean * total_images + batch_array.sum(axis=0)) / (total_images + args.batch_size)
            ipca.partial_fit(batch_array - mean)
            mmap_emb[:] = batch_array
            mmap_emb.flush()
            current_batch = current_batch[args.batch_size:]
            total_images += args.batch_size

    if current_batch:
        batch_array = np.array(current_batch)
        ipca.partial_fit(batch_array - mean)
        mmap_emb[:len(current_batch)] = batch_array
        total_images += len(current_batch)

    main_pbar.close()

    np.save(os.path.join(args.output_dir, "pca_mean.npy"), mean)
    np.save(os.path.join(args.output_dir, "pca_components.npy"), ipca.components_)
    np.save(os.path.join(args.output_dir, "pca_explained_variance.npy"), ipca.explained_variance_)

    del mmap_emb
    os.remove(mmap_path)

    print(f"\nPCA completed! Processed {total_images} images.")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
