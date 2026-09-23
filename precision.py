    # ===== METRICS =====
    THR = 0.5   # confidence threshold, change if you want
    conf = np.array(all_conf)
    correct = np.array(all_correct)
    total_gt = len(correct)

    tp = int(np.sum(correct & (conf >= THR)))
    fp = int(np.sum(~correct & (conf >= THR)))
    fn = total_gt - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / total_gt

    order = np.argsort(-conf)
    c = correct[order]
    ap = np.sum((np.cumsum(c) / np.arange(1, total_gt + 1)) * c) / total_gt

    print("Total GT  :", total_gt)
    print("TP        :", tp)
    print("FP        :", fp)
    print("FN        :", fn)
    print("Precision : %.2f%%" % (precision * 100))
    print("Recall    : %.2f%%" % (recall * 100))
    print("AP        : %.2f%%" % (ap * 100))
