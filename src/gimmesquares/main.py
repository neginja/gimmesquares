import click
import cv2
from cv2.typing import MatLike
import numpy as np
from pathlib import Path
from rembg import remove
from skimage.morphology import skeletonize


def quantize_colors(img: MatLike, n_colors: int = 8) -> MatLike:
    """Reduce number of colors using k-means."""
    data = img.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(
        data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    center = center.astype(np.uint8)
    quantized = center[label.flatten()].reshape(img.shape)
    return quantized


def get_dynamic_thresholds(
    channel: np.ndarray, sensitivity: float = 0.5, min_t: int = 5, max_t: int = 200
) -> tuple[int, int]:
    """
    Computes thresholds based on deviation from the neutral 128 center (for lab color space channels).
    """
    deviation = np.mean(np.abs(channel.astype(np.int16) - 128))

    lower = int(deviation * (1 - sensitivity) * 2)
    upper = int(deviation * (1 - sensitivity) * 5)

    # clamp
    return max(min_t, lower), min(max_t, upper)


def hex_to_bgr(hex_str: str) -> tuple[int, int, int]:
    """Converts #RRGGBB to (B, G, R)."""
    hex_str = hex_str.lstrip("#")
    lv = len(hex_str)
    rgb = tuple(int(hex_str[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return (rgb[2], rgb[1], rgb[0])


def mask_to_bgra(mask: MatLike, fill_color: tuple[int, int, int]) -> MatLike:
    h, w = mask.shape
    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    bgra[:, :, :3] = fill_color  # fill whole picture
    bgra[:, :, 3] = mask  # set the mask as alpha
    return bgra


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--pixel-size", default=8, type=int, help="Size of the 'pixels'.")
@click.option(
    "--colors", default=16, type=int, help="Number of colors for auto-palette."
)
@click.option("--bg/--no-bg", is_flag=True, default=True, help="Remove background.")
@click.option(
    "--alpha-matting", is_flag=True, default=True, help="Use alpha matting in rembg."
)
@click.option(
    "--af-thresh", default=240, type=int, help="Foreground threshold for matting."
)
@click.option(
    "--ab-thresh", default=10, type=int, help="Background threshold for matting."
)
@click.option(
    "--outline/--no-outline", is_flag=True, default=False, help="Draw contours."
)
@click.option("--outline-color", default="#000000", help="Hex color of the outline.")
@click.option(
    "--outline-sensitivity",
    default=0.5,
    type=float,
    help="Sensitivity of the captured outlines, higher, the more sensitive, range 0 to 1.",
)
@click.option(
    "--k-size",
    default=11,
    type=int,
    help="Kernel size of the blur before contours detection used to draw the outline (negative size to disable).",
)
@click.option(
    "--gaussian/--no-gaussian",
    is_flag=True,
    default=False,
    help="Use gaussian or median filter before contour detection.",
)
@click.option(
    "--save-extras/--no-save-extras",
    is_flag=True,
    default=False,
    help="Whether or not to save outlines and pixelized outputs aside from the combination.",
)
def main(
    input_path: str | Path,
    output_path: str | Path,
    pixel_size: int,
    colors: int,
    bg: bool,
    alpha_matting: bool,
    af_thresh: int,
    ab_thresh: int,
    outline: bool,
    outline_sensitivity: float,
    k_size: int,
    gaussian: bool,
    outline_color: str,
    save_extras: bool,
) -> None:
    """Convert images to pixel art using OpenCV."""
    input_path: Path = Path(input_path)
    output_path: Path = Path(output_path)
    if not input_path.exists() and input_path.stem.strip(".").lower() in [
        "jpg",
        "jpeg",
        "png",
        "webp",
    ]:
        click.echo("Error: Invalid input image...")
        return

    source_img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if source_img is None:
        click.echo("Error: Could not read image.")
        return
    img = source_img.copy()

    if not bg:
        click.echo("Info: Removing background...")
        img = remove(
            img,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=af_thresh,
            alpha_matting_background_threshold=ab_thresh,
            post_process_mask=True,
        )

    h, w = img.shape[:2]
    small_w = max(1, w // pixel_size)
    small_h = max(1, h // pixel_size)

    # downscale
    img_small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_LANCZOS4)

    # separate alpha if it exists since it doesn't influence the quantization
    has_alpha = img_small.shape[2] == 4
    if has_alpha:
        small_img_bgr = img_small[:, :, :3]
        small_img_alpha = img_small[:, :, 3]
    else:
        small_img_bgr = img_small
        small_img_alpha = None

    click.echo(f"Info: Quantizing to {colors} colors...")
    pixelated_bgr = quantize_colors(small_img_bgr, n_colors=colors)

    # keep a copy if needed to be saved un-altered
    pixelated_bgr_orig = pixelated_bgr.copy()
    if has_alpha:
        pixelated_bgr_orig = cv2.merge(
            [*cv2.split(pixelated_bgr_orig), small_img_alpha]
        )

    if outline:
        bgr_color = hex_to_bgr(outline_color)
        # work in l,a,b color space to detect outlines
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        _, a, b = cv2.split(lab)

        if k_size > 0:
            if gaussian:
                a = cv2.GaussianBlur(a, (k_size, k_size), 0)
                b = cv2.GaussianBlur(b, (k_size, k_size), 0)
            else:
                a = cv2.medianBlur(a, k_size)
                b = cv2.medianBlur(b, k_size)

        lower_a, higher_a = get_dynamic_thresholds(a, outline_sensitivity)
        lower_b, higher_b = get_dynamic_thresholds(b, outline_sensitivity)

        edges_a = cv2.Canny(a, lower_a, higher_a)
        edges_b = cv2.Canny(b, lower_b, higher_b)

        combined_edges = cv2.bitwise_or(edges_a, edges_b)

        if img.shape[2] == 4:
            # if alpha channel also threshold it, this is the case when background removal is in action
            # and it helps draw the object outer contour
            hr_alpha = img[:, :, 3]
            lower_alpha, higher_alpha = get_dynamic_thresholds(
                hr_alpha, outline_sensitivity
            )
            alpha_edges = cv2.Canny(hr_alpha, lower_alpha, higher_alpha)
            combined_edges = cv2.bitwise_or(combined_edges, alpha_edges)

        # since we detect edges from two channels, close gaps between edges and skeletonize
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, close_k)

        # reshape full res contours into a (small_h, pixel_size, small_w, pixel_size) mat
        reshaped = combined_edges[
            : small_h * pixel_size, : small_w * pixel_size
        ].reshape(small_h, pixel_size, small_w, pixel_size)

        # perform max pooling over the 2nd and 4th axes (the block dimensions)
        small_edges = reshaped.max(axis=(1, 3))
        # skeletonize the end result to ensure smooth tiny outlines
        small_edges = skeletonize(small_edges)
        small_edges = small_edges.astype(np.uint8) * 255

        if save_extras:
            outline_path = (
                output_path.parent / f"{output_path.stem}_outlines{output_path.suffix}"
            )
            # upscale outlines to save them as extras
            outlines = cv2.resize(small_edges, (w, h), interpolation=cv2.INTER_NEAREST)
            outlines_w_transparency = mask_to_bgra(outlines, fill_color=bgr_color)
            click.echo(f"Info: Saving the outlines extras to {outline_path}..")
            _ = cv2.imwrite(outline_path, outlines_w_transparency)

        # draw outline on a copy
        pixelated_bgr[small_edges > 0] = bgr_color
        if small_img_alpha is not None:
            small_img_alpha[small_edges > 0] = (
                255  # ensure contour is always fully opaque
            )

    if small_img_alpha is not None:
        final_pixelated = cv2.merge([*cv2.split(pixelated_bgr), small_img_alpha])
    else:
        final_pixelated = pixelated_bgr

    # upscale final result
    result = cv2.resize(final_pixelated, (w, h), interpolation=cv2.INTER_NEAREST)

    if save_extras:
        # also save the pixelated image without the contours
        pixelated_hr = cv2.resize(
            pixelated_bgr_orig, (w, h), interpolation=cv2.INTER_NEAREST
        )
        pixelated_path = (
            output_path.parent / f"{output_path.stem}_pixelated{output_path.suffix}"
        )
        click.echo(f"Info: Saving the pixelated extras to {pixelated_path}..")
        _ = cv2.imwrite(pixelated_path, pixelated_hr)

    _ = cv2.imwrite(output_path, result)

    click.echo(f"Info: Successfully saved result to {output_path}")


if __name__ == "__main__":
    main()
