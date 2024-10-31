import os
SUPPORT_FORMAT = (".jpg", ".jpeg", ".png",".bmp")

def load_lbls(anno_dir) -> dict:
    """Load lbls into dict

    Args:
        anno_dir (str): dir to annotation path with .txt files

    Returns:
        dict: key: anno_path value: lbls, [[id, xc,yc,wn,hn],...]
    """
    lbls = {}
    for file in os.listdir(anno_dir):
        if file.endswith('.txt'):
            anno_path = os.path.join(anno_dir, file)
            with open(anno_path, 'r') as f:
                lbls[anno_path] = [ line.strip().split(" ") for line in f.readlines()]
    return lbls

def load_detect_img_paths(data_path) -> list :
    return [ os.path.join(data_path, file) for file in os.listdir(data_path) if file.lower().endswith(SUPPORT_FORMAT)]

def load_classify_img_paths(data_path) -> list :
    return [os.path.join(cur_root, file) for cur_root, _, files in os.walk(data_path) for file in files if file.lower().endswith(SUPPORT_FORMAT)]

if __name__ == "__main__":
    lbls = load_lbls(r".\datasets\infer_trial_MIS_crop\labels")
    print(lbls)