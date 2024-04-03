import argparse


def parser():
    parser = argparse.ArgumentParser(description="attack")

    parser.add_argument(
        "--todo", choices=["train", "test"], default="train", help="train|test"
    )
    parser.add_argument("--device", default="0", type=str, help="0123")

    parser.add_argument("--dataset", default="face_dataset")
    parser.add_argument("--dataset2", default=None)
    parser.add_argument("--epoches", type=int, default=10)
    parser.add_argument("--save_each_epoch", type=int, default=5)
    parser.add_argument("--save_path", default="face1")
    parser.add_argument("--lr", default=5e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--load_path", default=None)
    parser.add_argument("--load_epoch", type=int, default=0)

    parser.add_argument("--adv_train", type=bool, default=False)
    parser.add_argument("--atk_eps", default=8 / 255)
    parser.add_argument("--atk_alpha", default=2 / 225)
    parser.add_argument("--atk_steps", type=int, default=10)

    return parser.parse_args()