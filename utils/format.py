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