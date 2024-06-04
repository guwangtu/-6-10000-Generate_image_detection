import argparse


def str_to_float(value_str):
    """
    将分数形式的字符串转换为浮点数。
    例如: "8/255" -> 0.03137254901960784
    """
    if "/" in value_str:
        numerator, denominator = map(float, value_str.split("/"))
        return numerator / denominator
    else:
        return float(value_str)


def parser():
    parser = argparse.ArgumentParser(description="attack")

    parser.add_argument(
        "--todo",
        choices=["train", "test", "get_adv_imgs"],
        default="train",
        help="train|test|get_adv_imgs",
    )
    parser.add_argument("--device", default="0", type=str, help="0123")

    parser.add_argument("--model", default="resnet", type=str, help="resnet,vit")

    parser.add_argument("--dataset", default="face_dataset")
    parser.add_argument("--train_dataset2", default=None)
    parser.add_argument("--val_dataset2", default=None)
    parser.add_argument("--epoches", type=int, default=10)
    parser.add_argument("--adv_test", default=False, action="store_true")
    parser.add_argument("--save_each_epoch", type=int, default=5)
    parser.add_argument("--save_path", default="face1")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--load_path", default=None)
    parser.add_argument("--load_epoch", type=int, default=0)
    parser.add_argument("--sgd", default=False, action="store_true")

    parser.add_argument("--adv", default=False, action="store_true")
    parser.add_argument("--atk_eps", type=str_to_float, default=8 / 255)
    parser.add_argument("--atk_alpha", type=str_to_float, default=2 / 255)
    parser.add_argument("--atk_steps", type=int, default=10)
    parser.add_argument("--update_adv_each_epoch", type=int, default=100)

    parser.add_argument("--test_first", default=False, action="store_true")
    parser.add_argument("--artifact", default=False, action="store_true")
    parser.add_argument("--df", default=False, action="store_true")
    parser.add_argument("--genimage", default=False, action="store_true")
    parser.add_argument("--imagenet", default=None)

    #parser.add_argument("--log_path", default="log/training.log") #改成自动的了

    return parser.parse_args()
