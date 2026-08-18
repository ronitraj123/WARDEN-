import os
import re
import json
import base64

from google import genai
from dotenv import load_dotenv


# ============================================================
# Configuration
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable not set. "
        "Get a key from https://aistudio.google.com/apikey "
        "and set it before running."
    )


MODEL_NAME = "gemini-3.6-flash"

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# Indian plate validation
# ============================================================

INDIAN_PLATE_REGEX = re.compile(
    r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4}$'
)


# ============================================================
# Gemini prompt
# ============================================================

PROMPT = """
You are looking at a CCTV image of a car at a railway crossing.

Find the vehicle's license plate and read the text on it exactly
as printed.

Indian license plates generally follow this format:

- two letters: state code
- one or two digits: district/RTO code
- one to three letters: series
- three or four digits: registration number

Examples:

PB11BV2245
MH12AB1234
DL01AB1234

Important instructions:

1. Read only the vehicle registration plate.
2. Do not use surrounding text, stickers, logos, or vehicle markings.
3. Do not invent or guess characters.
4. If a character is unclear, do not guess it.
5. If the plate cannot be reliably read, return null.
6. Remove spaces and hyphens from the plate text.

Respond with ONLY a JSON object in exactly this format:

{
  "plate_text": "<plate text with no spaces, or null if not readable>",
  "confidence": "<high, medium, or low>",
  "reasoning": "<one short sentence explaining what you saw>"
}
"""


# ============================================================
# Text cleaning
# ============================================================

def clean_text(text):

    if not text:
        return None

    return (
        str(text)
        .upper()
        .replace(" ", "")
        .replace("-", "")
    )


# ============================================================
# Indian plate format validation
# ============================================================

def strict_format(text):

    return (
        bool(text)
        and bool(
            INDIAN_PLATE_REGEX.fullmatch(text)
        )
    )


# ============================================================
# Gemini OCR
# ============================================================

def read_plate_with_gemini(image_path):
    """
    Send one selected vehicle snapshot to Gemini.

    Parameters
    ----------
    image_path : str
        Path of the snapshot selected from the dashboard.

    Returns
    -------
    dict
        {
            "plate_text": str or None,
            "confidence": str,
            "reasoning": str,
            "valid_format": bool
        }
    """

    # --------------------------------------------------------
    # Validate image path
    # --------------------------------------------------------

    if not image_path:
        raise ValueError(
            "No image path provided."
        )

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # --------------------------------------------------------
    # Determine MIME type
    # --------------------------------------------------------

    ext = os.path.splitext(
        image_path
    )[1].lower()

    if ext in (".jpg", ".jpeg"):
        mime_type = "image/jpeg"

    elif ext == ".png":
        mime_type = "image/png"

    else:
        raise ValueError(
            f"Unsupported image format: {ext}"
        )

    # --------------------------------------------------------
    # Encode image
    # --------------------------------------------------------

    image_b64 = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    # --------------------------------------------------------
    # Send image to Gemini
    # --------------------------------------------------------

    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=[
            {
                "type": "image",
                "data": image_b64,
                "mime_type": mime_type,
            },
            {
                "type": "text",
                "text": PROMPT,
            },
        ],
    )

    # --------------------------------------------------------
    # Read Gemini response
    # --------------------------------------------------------

    raw = interaction.output_text.strip()

    # Remove Markdown code fences if Gemini returns them.

    if raw.startswith("```"):

        raw = re.sub(
            r"^```(?:json)?",
            "",
            raw,
            flags=re.IGNORECASE,
        )

        raw = re.sub(
            r"```$",
            "",
            raw,
        ).strip()

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        parsed = json.loads(raw)

    except json.JSONDecodeError:

        print(
            "WARNING: Could not parse Gemini response:"
        )

        print(raw)

        return {
            "plate_text": None,
            "confidence": "low",
            "reasoning": "Unparseable Gemini response.",
            "valid_format": False,
        }

    # --------------------------------------------------------
    # Extract fields
    # --------------------------------------------------------

    plate_text = clean_text(
        parsed.get("plate_text")
    )

    confidence = str(
        parsed.get(
            "confidence",
            "low"
        )
    ).lower()

    reasoning = str(
        parsed.get(
            "reasoning",
            ""
        )
    )

    # --------------------------------------------------------
    # Validate Indian plate format
    # --------------------------------------------------------

    valid = strict_format(
        plate_text
    )

    # Only return the plate if it satisfies
    # the Indian format.

    if not valid:
        plate_text = None

    # --------------------------------------------------------
    # Return structured result
    # --------------------------------------------------------

    return {
        "plate_text": plate_text,
        "confidence": confidence,
        "reasoning": reasoning,
        "valid_format": valid,
    }