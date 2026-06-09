import cv2
import numpy as np
import argparse


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array([
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16)
    ], dtype=np.uint8)


def zhang_suen_thinning(binary):
    """
    Skeletonize a binary mask to make lines finer.
    Input: binary mask with values 0 or 255.
    Output: thinned binary mask, 0 or 255.
    """
    img = binary.copy()
    img[img > 0] = 1

    changed = True
    while changed:
        changed = False
        to_remove = []

        rows, cols = img.shape

        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                p1 = img[y, x]
                if p1 != 1:
                    continue

                p2 = img[y - 1, x]
                p3 = img[y - 1, x + 1]
                p4 = img[y, x + 1]
                p5 = img[y + 1, x + 1]
                p6 = img[y + 1, x]
                p7 = img[y + 1, x - 1]
                p8 = img[y, x - 1]
                p9 = img[y - 1, x - 1]

                neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                bp = sum(neighbors)

                transitions = 0
                circular = neighbors + [neighbors[0]]
                for i in range(8):
                    if circular[i] == 0 and circular[i + 1] == 1:
                        transitions += 1

                if (
                    2 <= bp <= 6 and
                    transitions == 1 and
                    p2 * p4 * p6 == 0 and
                    p4 * p6 * p8 == 0
                ):
                    to_remove.append((y, x))

        if to_remove:
            changed = True
            for y, x in to_remove:
                img[y, x] = 0

        to_remove = []

        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                p1 = img[y, x]
                if p1 != 1:
                    continue

                p2 = img[y - 1, x]
                p3 = img[y - 1, x + 1]
                p4 = img[y, x + 1]
                p5 = img[y + 1, x + 1]
                p6 = img[y + 1, x]
                p7 = img[y + 1, x - 1]
                p8 = img[y, x - 1]
                p9 = img[y - 1, x - 1]

                neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                bp = sum(neighbors)

                transitions = 0
                circular = neighbors + [neighbors[0]]
                for i in range(8):
                    if circular[i] == 0 and circular[i + 1] == 1:
                        transitions += 1

                if (
                    2 <= bp <= 6 and
                    transitions == 1 and
                    p2 * p4 * p8 == 0 and
                    p2 * p6 * p8 == 0
                ):
                    to_remove.append((y, x))

        if to_remove:
            changed = True
            for y, x in to_remove:
                img[y, x] = 0

    return (img * 255).astype(np.uint8)


def catmull_rom_spline(points, samples_per_segment=25, closed=True):
    """
    Smooth curve through control points.
    points should be Nx2.
    """
    points = np.array(points, dtype=np.float32)

    if closed:
        pts = np.vstack([points[-1], points, points[0], points[1]])
    else:
        pts = np.vstack([points[0], points, points[-1]])

    curve = []

    for i in range(1, len(pts) - 2):
        p0 = pts[i - 1]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[i + 2]

        for j in range(samples_per_segment):
            t = j / samples_per_segment
            t2 = t * t
            t3 = t2 * t

            point = 0.5 * (
                (2 * p1) +
                (-p0 + p2) * t +
                (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            curve.append(point)

    return np.array(curve, dtype=np.int32)


def draw_dotted_polyline(img, points, color=(0, 0, 255), radius=1, step=7):
    """
    Draw dotted line using BGR color.
    """
    for i in range(0, len(points), step):
        x, y = points[i]
        cv2.circle(img, (int(x), int(y)), radius, color, -1)


def normalized_to_pixel(points_norm, width, height):
    points = []
    for x, y in points_norm:
        points.append((int(x * width), int(y * height)))
    return points


def process_image(
    input_path,
    output_path,
    crop_box=(72, 28, 300, 500),
    lcfs_hex="#FA0580",
    lcfs_tolerance=65,
    mode="overlay",
    lcfs_width=1,
    dot_radius=1,
    dot_step=7
):
    image_bgr = cv2.imread(input_path)

    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    left, top, right, bottom = crop_box
    crop_bgr = image_bgr[top:bottom, left:right]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    h, w, _ = crop_bgr.shape

    if mode == "binary":
        output = np.zeros_like(crop_bgr)
    else:
        output = crop_bgr.copy()

    # --------------------------------------------------
    # 1. Fine LCFS extraction from pink color
    # --------------------------------------------------
    lcfs_rgb = hex_to_rgb(lcfs_hex)

    color_dist = np.linalg.norm(
        crop_rgb.astype(np.int16) - lcfs_rgb.astype(np.int16),
        axis=2
    )

    lcfs_mask = (color_dist < lcfs_tolerance).astype(np.uint8) * 255

    # Clean tiny gaps/noise gently
    lcfs_mask = cv2.morphologyEx(
        lcfs_mask,
        cv2.MORPH_CLOSE,
        np.ones((2, 2), np.uint8),
        iterations=1
    )

    # Make LCFS fine instead of thick
    lcfs_mask = zhang_suen_thinning(lcfs_mask)

    if lcfs_width > 1:
        lcfs_mask = cv2.dilate(
            lcfs_mask,
            np.ones((lcfs_width, lcfs_width), np.uint8),
            iterations=1
        )

    # White LCFS
    output[lcfs_mask > 0] = (255, 255, 255)

    # --------------------------------------------------
    # 2. Manually controlled D templates
    #
    # These are normalized coordinates inside the crop.
    # x and y range from 0 to 1.
    # You can tune these once and reuse for the full dataset
    # if the screenshots have the same plot layout.
    # --------------------------------------------------

    outer_d_norm = [
        (0.10, 0.08),
        (0.28, 0.00),
        (0.50, 0.03),
        (0.70, 0.18),
        (0.85, 0.40),
        (0.88, 0.62),
        (0.77, 0.82),
        (0.55, 0.96),
        (0.30, 1.00),
        (0.12, 0.90),
        (0.03, 0.68),
        (0.02, 0.42),
    ]

    inner_d_norm = [
        (0.30, 0.25),
        (0.43, 0.23),
        (0.55, 0.32),
        (0.68, 0.44),
        (0.70, 0.60),
        (0.62, 0.73),
        (0.46, 0.83),
        (0.28, 0.77),
        (0.20, 0.60),
        (0.20, 0.40),
    ]

    outer_d_px = normalized_to_pixel(outer_d_norm, w, h)
    inner_d_px = normalized_to_pixel(inner_d_norm, w, h)

    outer_curve = catmull_rom_spline(outer_d_px, samples_per_segment=30, closed=True)
    inner_curve = catmull_rom_spline(inner_d_px, samples_per_segment=30, closed=True)

    # Red dotted D lines, BGR red = (0, 0, 255)
    draw_dotted_polyline(
        output,
        outer_curve,
        color=(0, 0, 255),
        radius=dot_radius,
        step=dot_step
    )

    draw_dotted_polyline(
        output,
        inner_curve,
        color=(0, 0, 255),
        radius=dot_radius,
        step=dot_step
    )

    cv2.imwrite(output_path, output)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="lcfs_sol_overlay.png")

    parser.add_argument("--left", type=int, default=72)
    parser.add_argument("--top", type=int, default=28)
    parser.add_argument("--right", type=int, default=300)
    parser.add_argument("--bottom", type=int, default=500)

    parser.add_argument("--lcfs_hex", default="#FA0580")
    parser.add_argument("--tolerance", type=int, default=65)

    parser.add_argument(
        "--mode",
        choices=["overlay", "binary"],
        default="overlay",
        help="overlay = draw on cropped original image, binary = black background"
    )

    parser.add_argument("--lcfs_width", type=int, default=1)
    parser.add_argument("--dot_radius", type=int, default=1)
    parser.add_argument("--dot_step", type=int, default=7)

    args = parser.parse_args()

    process_image(
        input_path=args.input,
        output_path=args.output,
        crop_box=(args.left, args.top, args.right, args.bottom),
        lcfs_hex=args.lcfs_hex,
        lcfs_tolerance=args.tolerance,
        mode=args.mode,
        lcfs_width=args.lcfs_width,
        dot_radius=args.dot_radius,
        dot_step=args.dot_step
    )
