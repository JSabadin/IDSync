import argparse
import importlib.util
import sys
from src.fr_model import train_fr
from src.fr_model.export import export_model_to_onnx
from src.atribute_model import train_atribute

def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    if hasattr(config_module, 'config'):
        return config_module.config
    raise ValueError("The config file does not define 'config'.")

def main():
    parser = argparse.ArgumentParser(prog="ml-tool", description="CLI for all ML workflows")
    subparsers = parser.add_subparsers(dest="model_type", required=True, help="Model type to operate on")

    # face_recognition
    fr = subparsers.add_parser("face_recognition", help="Face recognition tasks")
    fr_sub = fr.add_subparsers(dest="action", required=True, help="Actions for face_recognition")
    for act in ("train", "evaluate"):
        p = fr_sub.add_parser(act, help=f"{act} a face recognition model")
        p.add_argument("--config", required=True, help="Path to config.py")
    export = fr_sub.add_parser("export", help="Export FR model to ONNX")
    export.add_argument("--config", required=True, help="Path to config.py")

    # attribute_model
    am = subparsers.add_parser("attribute_model", help="Attribute model tasks")
    am_sub = am.add_subparsers(dest="action", required=True, help="Actions for attribute_model")
    for act in ("train", "evaluate"):
        p = am_sub.add_parser(act, help=f"{act} an attribute model")
        p.add_argument("--config", required=True, help="Path to config.py")

    # diffusion
    sd = subparsers.add_parser("diffusion", help="Stable Diffusion commands")
    sd_sub = sd.add_subparsers(dest="action", required=True, help="Diffusion sub-commands")

    # run_pca
    pca = sd_sub.add_parser("run_pca", help="Run PCA on ArcFace embeddings")
    pca.add_argument("--arcface_onnx_path", required=True, help="Path to ArcFace ONNX model")
    pca.add_argument("--dataset_root", required=True, help="Root directory of dataset with subfolders per identity")
    pca.add_argument("--output_dir", required=True, help="Directory to store PCA outputs")
    pca.add_argument("--json_mapping", required=True, help="Path to JSON mapping file (e.g., webface21_mapping.json)")
    pca.add_argument("--batch_size", type=int, default=10000, help="Batch size for incremental PCA")
    pca.add_argument("--pca_components", type=int, default=400, help="Number of PCA components")


    # precompute
    pre = sd_sub.add_parser("precompute", help="Precompute T2I embeddings")
    pre.add_argument("--arcface_onnx", default=None)
    pre.add_argument("--base_model", default=None)
    pre.add_argument("--image_dir", default=None)
    pre.add_argument("--prompt_dir", default=None)

    # train subcommand (just a shell for dynamic help)
    train_shell = sd_sub.add_parser("train", add_help=False, help="Fine-tune Stable Diffusion")
    train_shell.set_defaults(is_train=True)
    train_shell.add_argument("-h", "--help", dest="train_help", action="store_true", help="Show train help and exit")

    # infer
    inf_shell = sd_sub.add_parser("infer", add_help=False, help="Run inference with Stable Diffusion")
    inf_shell.set_defaults(is_infer=True)
    inf_shell.add_argument("-h", "--help", dest="infer_help", action="store_true", help="Show infer help and exit")

    # create_synth
    synth_shell = sd_sub.add_parser("create_synth", add_help=False, help="Create synthetic dataset")
    synth_shell.set_defaults(is_synth=True)
    synth_shell.add_argument("-h", "--help", dest="synth_help", action="store_true", help="Show create_synth help and exit")


    args, remaining = parser.parse_known_args()

    # Handle diffusion/train
    if getattr(args, "is_train", False):
        from src.diffusion_model.train import parse_args as parse_train_args, train_sd
        if args.train_help or "-h" in remaining or "--help" in remaining:
            tmp_parser = argparse.ArgumentParser(prog="ml-tool diffusion train", description="Training arguments")
            from src.diffusion_model.train import add_train_args
            add_train_args(tmp_parser)
            tmp_parser.print_help()
            return
        train_args = parse_train_args(remaining)
        train_sd(train_args)
        return

    # Handle diffusion/infer
    if getattr(args, "is_infer", False):
        from src.diffusion_model.infer import parse_args as parse_infer_args, infer_sd
        if args.infer_help or "-h" in remaining or "--help" in remaining:
            tmp_parser = argparse.ArgumentParser(prog="ml-tool diffusion infer", description="Inference arguments")
            from src.diffusion_model.infer import add_infer_args
            add_infer_args(tmp_parser)
            tmp_parser.print_help()
            return
        infer_args = parse_infer_args(remaining)
        infer_sd(infer_args)
        return

    if getattr(args, "is_synth", False):
        from src.diffusion_model.create_synth_dataset import add_synth_args, parse_args as _parse_synth, create_synth_dataset
        # help
        if args.synth_help or "-h" in remaining or "--help" in remaining:
            tmp = argparse.ArgumentParser(prog="ml-tool diffusion create_synth", description="Synthetic dataset arguments")
            add_synth_args(tmp)
            tmp.print_help()
            return
        synth_args = _parse_synth(remaining)
        create_synth_dataset(synth_args)
        return

    # dispatch for other subcommands
    if args.model_type == "face_recognition":
        cfg = load_config(args.config)
        if args.action == "train":
            train_fr(cfg)
        elif args.action == "evaluate":
            cfg['trainer']['only_validate'] = True
            train_fr(cfg)
        elif args.action == "export":
            export_model_to_onnx(load_config(args.config))
            print("Model exported to ONNX successfully")

    elif args.model_type == "attribute_model":
        cfg = load_config(args.config)
        if args.action == "train":
            train_atribute(cfg)
        elif args.action == "evaluate":
            cfg['trainer']['only_validate'] = True
            train_atribute(cfg)

    elif args.model_type == "diffusion":
        if args.action == "precompute":
            from src.diffusion_model.precompute_dataset import (
                precompute_dataset, parse_args as _parse
            )
            pre_args = _parse(sys.argv[sys.argv.index("precompute") + 1:])
            precompute_dataset(pre_args)
        elif args.action == "infer":
            from src.diffusion_model.infer import infer_sd, parse_args as _parse
            infer_sd(_parse(remaining))
        elif args.action == "run_pca":
            from src.diffusion_model.pca_pipeline import run_pca_main
            run_pca_main(args)

if __name__ == "__main__":
    main()