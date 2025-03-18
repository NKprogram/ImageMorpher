import os
import requests
import random
from flask import Flask, request, jsonify
from mangum import Mangum
from asgiref.wsgi import WsgiToAsgi
from dotenv import load_dotenv
import nacl.signing
import nacl.exceptions



from ImageConversion import (
    convert_image_to_ascii,
    convert_image_to_emoji,
    convert_image_to_pixel_art,
    convert_image_to_blur,
    convert_image_to_deep_fry,
    convert_image_to_sketch,
    convert_image_to_oil_paint,
    convert_image_to_watercolor,
    convert_image_to_cartoon,
    convert_image_to_glitch,
    convert_image_to_neon_glow,
    convert_image_to_pop_art,
    convert_image_to_mosaic,
    convert_image_to_sepia
)

load_dotenv()

DISCORD_PUBLIC_KEY = os.environ.get("DISCORD_PUBLIC_KEY")  
APPLICATION_ID = os.environ.get("APPLICATION_ID")           
BOT_TOKEN = os.environ.get("DISCORD_TOKEN")                 

app = Flask(__name__)
asgi_app = WsgiToAsgi(app)
handler = Mangum(asgi_app)


# Verify the request signature
def verify_signature(req):
    signature = req.headers.get("X-Signature-Ed25519")
    timestamp = req.headers.get("X-Signature-Timestamp")
    body = req.get_data(as_text=True) 
    if not signature or not timestamp:
        return False
    try:
        verify_key = nacl.signing.VerifyKey(bytes.fromhex(DISCORD_PUBLIC_KEY))
        verify_key.verify(f"{timestamp}{body}".encode(), bytes.fromhex(signature))
        return True
    except nacl.exceptions.BadSignatureError:
        return False
    


@app.route("/", methods=["POST"])
def interactions():
    if not verify_signature(request):
        return "invalid request signature", 401
    # Parse the incoming interaction request
    raw_request = request.json
    print(f"👉 Incoming interaction: {raw_request}")
    # If it's a PING, respond with PONG
    if raw_request["type"] == 1:
        return jsonify({"type": 1})  # PONG
    # Otherwise, it's an ApplicationCommand or other interaction
    if raw_request["type"] == 2:  # APPLICATION_COMMAND
        # type=5 => DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE
        deferred_response = {
            "type": 5,
            "data": {
                "content": "Working on your image... please wait!"
            }
        }
        # Send the deferred response immediately
        from threading import Thread
        t = Thread(target=handle_slash_command, args=(raw_request,))
        t.start()

        return jsonify(deferred_response)

    # Fallback if something else:
    return jsonify({"error": "Unhandled interaction type"}), 400


def handle_slash_command(interaction_data):
    # Extract the command name and token
    command_name = interaction_data["data"]["name"]
    interaction_token = interaction_data["token"]
    # Extract the attachment info and options
    attachments_info = interaction_data["data"].get("resolved", {}).get("attachments", {})
    options = interaction_data["data"].get("options", [])

    # Handle simple commands
    if command_name in ["intro", "help"]:
        # Just send a follow-up message with text
        followup_content = handle_simple_command(command_name)
        create_followup_message(interaction_token, followup_content)
        return

    # otherwise, we need an attachment, as we are processing an image
    if not options or len(options) == 0:
        create_followup_message(interaction_token, "No attachment found in options.")
        return

    attachment_id = options[0]["value"]
    if attachment_id not in attachments_info:
        create_followup_message(interaction_token, "Unable to find attachment data.")
        return

    image_url = attachments_info[attachment_id]["url"]
    filename = attachments_info[attachment_id]["filename"]

    # Download the image
    local_input_path = f"/tmp/{filename}"  
    r = requests.get(image_url)
    with open(local_input_path, "wb") as f:
        f.write(r.content)

    # Prepare the output path
    base, ext = os.path.splitext(filename)
    suffix = command_name.replace("-", "")  # e.g. "deep-fry" -> "deepfry"
    local_output_path = f"/tmp/{base}_{suffix}{ext}"

    # Apply the correct transformation
    converted_path = process_image(command_name, local_input_path, local_output_path)

    if not converted_path:
        create_followup_message(interaction_token, f"Conversion for {command_name} failed.")
        return

    # Send the converted image back
    create_followup_file(interaction_token, converted_path)


def handle_simple_command(command_name):
    # Handle simple commands that don't require image processing
    if command_name == "intro":
        return (
            "**Hello there!** 👋 I'm the **Great Image Morpher**, ready to transform your images with a bit of slash-command magic.\n\n"
            "**✨ How to use:**\n"
            "Give my commands a try—just **attach an image** and let the enchantment begin!\n"
            "Use `/help` to see a list of my available transformations!"
        )
    
    elif command_name == "help":
        return (
            "**Here are my enchanted commands available to you.** **Use them well, mortal.**\n\n"
            "`/intro`\n"
            "‣ *Description of the bot and what it can do.*\n\n"
            "`/help`\n"
            "‣ *Get the list of available commands and descriptions.*\n\n"
            "`/ascii`\n"
            "‣ *Apply an ASCII-style transformation.*\n\n"
            "`/emoji`\n"
            "‣ *Convert into an emoji-based mosaic.*\n\n"
            "`/pixel`\n"
            "‣ *Transform into pixel art.*\n\n"
            "`/blurry`\n"
            "‣ *Blur the image.*\n\n"
            "`/deep-fry`\n"
            "‣ *Give a deep-fried look.*\n\n"
            "`/sketch`\n"
            "‣ *Convert into a pencil sketch.*\n\n"
            "`/oil-paint`\n"
            "‣ *Apply an oil paint effect.*\n\n"
            "`/watercolor`\n"
            "‣ *Transform into watercolor-style art.*\n\n"
            "`/cartoon`\n"
            "‣ *Give a cartoon-like appearance.*\n\n"
            "`/glitch`\n"
            "‣ *Apply a glitch effect.*\n\n"
            "`/neon-glow`\n"
            "‣ *Add a neon glow effect.*\n\n"
            "`/pop-art`\n"
            "‣ *Transform into pop art style.*\n\n"
            "`/mosaic`\n"
            "‣ *add a mosaicish blur.*\n\n"
            "`/sepia`\n"
            "‣ *Apply a sepia filter.*"
        )
    
    else:
        return "**Unknown command.** Use `/help` to see the available commands."


def process_image(command_name, input_path, output_path):
    # Process the image based on the command using the ImageConversion module
    try:
        if command_name == "ascii":
            return convert_image_to_ascii(input_path, output_path)
        elif command_name == "emoji":
            return convert_image_to_emoji(input_path, output_path)
        elif command_name == "pixel":
            return convert_image_to_pixel_art(input_path, output_path)
        elif command_name == "blurry":
            return convert_image_to_blur(input_path, output_path)
        elif command_name == "deep-fry":
            return convert_image_to_deep_fry(input_path, output_path)
        elif command_name == "sketch":
            return convert_image_to_sketch(input_path, output_path)
        elif command_name == "oil-paint":
            return convert_image_to_oil_paint(input_path, output_path)
        elif command_name == "watercolor":
            return convert_image_to_watercolor(input_path, output_path)
        elif command_name == "cartoon":
            return convert_image_to_cartoon(input_path, output_path)
        elif command_name == "glitch":
            return convert_image_to_glitch(input_path, output_path)
        elif command_name == "neon-glow":
            return convert_image_to_neon_glow(input_path, output_path)
        elif command_name == "pop-art":
            return convert_image_to_pop_art(input_path, output_path)
        elif command_name == "mosaic":
            return convert_image_to_mosaic(input_path, output_path)
        elif command_name == "sepia":
            return convert_image_to_sepia(input_path, output_path)
        else:
            return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def create_followup_message(interaction_token, message_content):
    # Create a follow-up message with text content
    url = f"https://discord.com/api/v10/webhooks/{APPLICATION_ID}/{interaction_token}"
    json_payload = {"content": message_content}
    headers = {"Authorization": f"Bot {BOT_TOKEN}"}
    requests.post(url, headers=headers, json=json_payload)

    

def create_followup_file(interaction_token, file_path):
    # Create a follow-up message with a file attachment
    url = f"https://discord.com/api/v10/webhooks/{APPLICATION_ID}/{interaction_token}"
    headers = {"Authorization": f"Bot {BOT_TOKEN}"}

    # random fun messages to send with the image
    messages = [
        "🖼️✨ Your image transformation is complete! Behold the magic! 🪄",
        "🚀 AI-powered makeover finished! Here’s your stunning new image. 🌟",
        "🔮 The digital sorcery is done—your image has a whole new vibe! 🎭",
        "🔥 Voila! Your transformed image is ready to dazzle. ✨",
        "🖌️ The pixels have been reshaped and reimagined—enjoy your masterpiece! 🎨",
        "🌟 A new look, an AI touch—your image is now one of a kind! 🔄",
        "⚡ Transformation complete! Your image just got a futuristic upgrade. 🚀",
        "📸 AI magic has worked wonders! Here’s your enhanced image. ✨"
    ]

    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "application/octet-stream")}
        data = {
            "content": random.choice(messages)  # Randomly selects a message
        }
        requests.post(url, headers=headers, data=data, files=files)

if __name__ == "__main__":
    app.run(debug=True)