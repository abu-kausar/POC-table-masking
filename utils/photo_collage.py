from PIL import Image, ImageDraw, ImageFont

def side_by_side_collage_with_caption(
    image_path_1: str,
    image_path_2: str,
    output_path: str = "collage_with_caption.jpg",
    caption_height: int = 100,
    font_size: int = 48
):
    img1 = Image.open(image_path_1).convert("RGB")
    img2 = Image.open(image_path_2).convert("RGB")

    # Resize to same height
    h = min(img1.height, img2.height)
    img1 = img1.resize((int(img1.width * h / img1.height), h), Image.LANCZOS)
    img2 = img2.resize((int(img2.width * h / img2.height), h), Image.LANCZOS)

    total_width = img1.width + img2.width
    total_height = h + caption_height

    collage = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(collage)

    # Load font
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    captions = ["Input Image", "Output Mask Image"]

    def text_size(text):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Caption positions
    text1_w, text1_h = text_size(captions[0])
    text2_w, text2_h = text_size(captions[1])

    text_y = (caption_height - text1_h) // 2
    text1_x = (img1.width - text1_w) // 2
    text2_x = img1.width + (img2.width - text2_w) // 2

    # Draw captions
    draw.text((text1_x, text_y), captions[0], fill=(0, 0, 0), font=font)
    draw.text((text2_x, text_y), captions[1], fill=(0, 0, 0), font=font)

    # Paste images
    collage.paste(img1, (0, caption_height))
    collage.paste(img2, (img1.width, caption_height))

    collage.save(output_path)
    print(f"✅ Collage saved at: {output_path}")


if __name__ == "__main__":
    side_by_side_collage_with_caption(
        image_path_1="src/images/ss-2.jpeg",
        image_path_2="src/masked_ss-2.png",
        output_path="result_collage.jpg"
    )
