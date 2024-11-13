def round_speeds(speeds):
    """Round speeds to two decimal places."""
    return [round(speed, 2) for speed in speeds]

def format_labels(labels):
    """Format label data for output. [x,y,x,y,conf,id]"""
    formatted = []
    for lbl in labels:
        formatted.extend(map(int, lbl[:-2]))
        formatted.append(round(lbl[-2], 3))
        formatted.append(int(lbl[-1]))
    return formatted

def format_confusion_matrix(metrics):
    class_ids_names = metrics.names
    detect_class_ids = metrics.box.ap_class_index
    summary = ''
    header = ("Class", "P", "R", "mAP50", "mAP50-95")
    header_format = f"\n{header[0]:<20} : " + " ".join(f"{h:>8}" for h in header[1:]) + "\n"
    summary += header_format
    all_result = metrics.results_dict
    summary += f"{'ALL':<20} : " + " ".join(f"{result:>8.4f}" for result in list(all_result.values())[:-1]) + "\n"
    for i, class_id in enumerate(detect_class_ids):
        class_result = metrics.box.class_result(i)
        formatted_results = list(map(lambda x: f"{x:>8.4f}", class_result))  
        summary += f"{class_ids_names[class_id]:<20} : {' '.join(formatted_results)}\n"
    return summary