from collections import defaultdict


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def match_detections(
    dets_a: list[dict],
    dets_b: list[dict],
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    pairs = []
    for i, da in enumerate(dets_a):
        for j, db in enumerate(dets_b):
            if da["class_id"] != db["class_id"]:
                continue
            iou = compute_iou(da["bbox_xyxy"], db["bbox_xyxy"])
            if iou >= iou_threshold:
                pairs.append((i, j, iou))

    pairs.sort(key=lambda x: x[2], reverse=True)

    used_a, used_b = set(), set()
    matched = []
    for i, j, _ in pairs:
        if i not in used_a and j not in used_b:
            matched.append((i, j))
            used_a.add(i)
            used_b.add(j)

    unmatched_a = [i for i in range(len(dets_a)) if i not in used_a]
    unmatched_b = [j for j in range(len(dets_b)) if j not in used_b]
    return matched, unmatched_a, unmatched_b


def _pick_best_det(candidates: list[dict]) -> dict:
    return max(candidates, key=lambda d: d["confidence"])


def flag_conflicts(
    per_model_results: dict[str, list[dict]],
    iou_threshold: float = 0.5,
) -> tuple[list[dict], list[dict]]:
    model_keys = list(per_model_results.keys())
    if len(model_keys) < 2:
        return list(per_model_results.values())[0], []

    images = {}
    for mk in model_keys:
        for item in per_model_results[mk]:
            key = item["image_path"]
            if key not in images:
                images[key] = {"width": item["width"], "height": item["height"], "models": {}}
            images[key]["models"][mk] = item["detections"]

    final_results = []
    review_results = []

    for img_path, info in images.items():
        all_dets = info["models"]
        has_conflict = False

        ref_key = model_keys[0]
        ref_dets = all_dets.get(ref_key, [])

        for other_key in model_keys[1:]:
            other_dets = all_dets.get(other_key, [])
            matched, unmatched_a, unmatched_b = match_detections(ref_dets, other_dets, iou_threshold)
            if unmatched_a or unmatched_b:
                has_conflict = True
                break

        if has_conflict:
            review_results.append({
                "image_path": img_path,
                "width": info["width"],
                "height": info["height"],
                "detections": ref_dets,
                "per_model": all_dets,
            })
        else:
            final_results.append({
                "image_path": img_path,
                "width": info["width"],
                "height": info["height"],
                "detections": ref_dets,
            })

    return final_results, review_results


def majority_vote(
    per_model_results: dict[str, list[dict]],
    iou_threshold: float = 0.5,
) -> tuple[list[dict], list[dict]]:
    model_keys = list(per_model_results.keys())
    n_models = len(model_keys)
    if n_models < 2:
        return list(per_model_results.values())[0], []

    majority = n_models // 2 + 1

    images = {}
    for mk in model_keys:
        for item in per_model_results[mk]:
            key = item["image_path"]
            if key not in images:
                images[key] = {"width": item["width"], "height": item["height"], "models": {}}
            images[key]["models"][mk] = item["detections"]

    final_results = []
    review_results = []

    for img_path, info in images.items():
        all_dets = info["models"]

        det_groups = []
        assigned = {mk: set() for mk in model_keys}

        for i, mk_a in enumerate(model_keys):
            for j in range(i + 1, len(model_keys)):
                mk_b = model_keys[j]
                dets_a = all_dets.get(mk_a, [])
                dets_b = all_dets.get(mk_b, [])
                matched, _, _ = match_detections(dets_a, dets_b, iou_threshold)

                for idx_a, idx_b in matched:
                    if idx_a in assigned[mk_a] and idx_b in assigned[mk_b]:
                        continue

                    merged = False
                    for group in det_groups:
                        if (mk_a, idx_a) in group or (mk_b, idx_b) in group:
                            group.add((mk_a, idx_a))
                            group.add((mk_b, idx_b))
                            assigned[mk_a].add(idx_a)
                            assigned[mk_b].add(idx_b)
                            merged = True
                            break
                    if not merged:
                        det_groups.append({(mk_a, idx_a), (mk_b, idx_b)})
                        assigned[mk_a].add(idx_a)
                        assigned[mk_b].add(idx_b)

        for mk in model_keys:
            dets = all_dets.get(mk, [])
            for idx in range(len(dets)):
                if idx not in assigned[mk]:
                    det_groups.append({(mk, idx)})
                    assigned[mk].add(idx)

        accepted = []
        has_total_disagreement = False

        for group in det_groups:
            models_in_group = {mk for mk, _ in group}
            candidates = [all_dets[mk][idx] for mk, idx in group]

            if len(models_in_group) >= majority:
                accepted.append(_pick_best_det(candidates))
            elif len(models_in_group) == 1 and n_models > 1:
                has_total_disagreement = True

        if has_total_disagreement and not accepted:
            review_results.append({
                "image_path": img_path,
                "width": info["width"],
                "height": info["height"],
                "detections": [],
                "per_model": all_dets,
            })
        else:
            if has_total_disagreement:
                review_results.append({
                    "image_path": img_path,
                    "width": info["width"],
                    "height": info["height"],
                    "detections": accepted,
                    "per_model": all_dets,
                })
            final_results.append({
                "image_path": img_path,
                "width": info["width"],
                "height": info["height"],
                "detections": accepted,
            })

    return final_results, review_results


def apply_consensus(
    per_model_results: dict[str, list[dict]],
    strategy: str,
    iou_threshold: float = 0.5,
) -> tuple[list[dict], list[dict]]:
    if strategy == "majority_vote":
        return majority_vote(per_model_results, iou_threshold)
    else:
        return flag_conflicts(per_model_results, iou_threshold)
