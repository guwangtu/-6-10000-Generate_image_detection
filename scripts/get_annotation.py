from load_data_artifact import get_annotation_artifact

import argparse

def main(args):
    if args.artifact:
        path='/data/user/shx/datasets/Artifact'
        s1='/data/user/shx/Generate_image_detection/LASTED/annotaion/train_artifact.txt'
        s2='/data/user/shx/Generate_image_detection/LASTED/annotaion/test_artifact.txt'
        get_annotation_artifact(path,s1,s2)
if __name__ == '__main__':
    conf = argparse.ArgumentParser()
    conf.add_argument("--artifact", default=False, action="store_true")
    args = conf.parse_args()
    main(args)