import os

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

if __name__ == "__main__":
    lbls = load_lbls(r".\datasets\infer_trial_MIS_crop\labels")
    print(lbls)