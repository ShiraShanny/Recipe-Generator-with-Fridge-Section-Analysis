import google.generativeai as genai
import streamlit as st
import ast

# gemmini


def detect_product_names(image_path):
    genai.configure(api_key="AIzaSyDtvZwuifgTdTfRg7VDfpG0iyd3OsFErRk")

    """Detects product names in an image using the Gemini API and returns them as a list.

    Args:
        image_path: Path to the image file.

    Returns:
        A list of detected product names or an empty list if none are found.
    """
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        image_blob = {
            "mime_type": "image/jpeg",  # Adjust this if using a different image format
            "data": image_bytes
        }

        # Initialize the model
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Define the request
        request = {
            "parts": [
                {"text": "Identify the individual products visible in this image. List each product separately."},
                {"inline_data": image_blob}
            ]
        }

        # Generate content using the Gemini model
        response = model.generate_content(request)

        # Extract the text part from the response
        if response.candidates:
            text_part = response.candidates[0].content.parts[0].text
            if text_part:
                # Print only the text part of the response
                st.write("Response details:", text_part)

                # Parse each line that starts with "*" (assumes products are listed this way)
                product_names = [item.strip("* ").strip() for item in text_part.splitlines() if
                                 item.strip().startswith("*")]
                return product_names
        else:
            st.write("No product names found in the response.")

    except Exception as e:
        print(f"An error occurred: {e}")

    return []



