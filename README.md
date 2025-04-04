# Imagemorpho - The Ultimate Image Morphing Bot!

## Overview
Imagemorpho is a Discord bot that transforms images into different styles like pixel art, sketches, glitches, and more. It can run on AWS Lambda for fast cloud processing, or you can run it locally if needed.

## Features
Imagemorpho can apply these effects:
- **ASCII**: Turns an image into ASCII art.
- **Emoji**: Builds an image out of emojis.
- **Pixel Art**: Pixelates the image.
- **Blurry**: Blurs the image.
- **Deep Fry**: Over-processes the image like a meme.
- **Sketch**: Turns the image into a pencil sketch.
- **Oil Paint**: Adds an oil paint effect.
- **Watercolor**: Makes it look like a watercolor painting.
- **Cartoon**: Adds a cartoon filter.
- **Glitch**: Creates a glitch effect.
- **Neon Glow**: Adds neon glow effects.
- **Pop Art**: Applies a pop art style.
- **Mosaic**: Breaks the image into a mosaic pattern.
- **Sepia**: Adds a sepia color tone.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip
- AWS Lambda set up (optional, for cloud deployment)
- Installed dependencies
- Ngrok account for local development

### Installation

1. Clone the repository.

2. Install the required dependencies:
   ```sh
   pip install -r commands/requirements.txt
   pip install -r src/app/requirements.txt

3. Set up your environment variables:
   ```sh
   export DISCORD_PUBLIC_KEY=your_public_key
   export APPLICATION_ID=your_application_id
   export DISCORD_TOKEN=your_bot_token
   ```
4 **Register the bots"" by running:
  ```sh
  python3 commands/register_commands.py
  ```

## Running Locally
1. Run the bot locally:
   ```sh
   python3 src/app/main.py
   ```

2. Create an [Ngrok](https://ngrok.com) account, download Ngrok, and authenticate your account.

3. Expose your local server to the internet using Ngrok Be sure to run this in a separate terminal window, apart from `main.py`:
   ```sh
   ngrok http 5000
   ```

4. Use the forwarding URL provided by Ngrok as your bot's endpoint URL in [Discord Developer Portal](https://discord.com/developers/applications)..

## Deploying to AWS Lambda
Imagemorpho is designed to run on **AWS Lambda** using **Mangum** for ASGI compatibility. Deploy using AWS CDK:

### Steps:
1. **Set up AWS credentials**  
   - Ensure you have an IAM user with the necessary permissions and configure the AWS CLI:  
     ```sh
     aws configure
     ```

2. Install **AWS CDK**:
   ```sh
   npm install -g aws-cdk
   ```

3. Bootstrap your AWS environment (only needed once per AWS account):
   ```sh
   cdk bootstrap aws://YOUR_ACCOUNT_ID/YOUR_REGION
   ```

4. Deploy the application using CDK:
   ```sh
   cdk deploy
   ```
5. After deployment, **CDK will output the endpoint URL** for your deployed **API Gateway**. Use this **URL as your bot's endpoint URL** in the [Discord Developer Portal](https://discord.com/developers/applications).

## How to Use

### Commands

Imagemorpho listens to these slash commands:

#### Informational Commands
- `/intro` - Explains what Imagemorpho is and what it does.
- `/help` - Shows a list of all transformations and how to use them.

#### Image Transformation Commands
- `/ascii` - Turns your image into ASCII art.
- `/emoji` - Builds a mosaic out of emojis using your image.
- `/pixel` - Pixelates your image.
- `/blurry` - Blurs your image.
- `/deep-fry` - Makes your image look deep-fried.
- `/sketch` - Converts your image into a pencil sketch.
- `/oil-paint` - Adds an oil paint effect.
- `/watercolor` - Makes your image look like a watercolor painting.
- `/cartoon` - Gives your image a cartoon style.
- `/glitch` - Adds glitch effects to your image.
- `/neon-glow` - Makes your image glow with neon lights.
- `/pop-art` - Changes your image into a pop-art style.
- `/mosaic` - Breaks your image into a mosaic pattern.
- `/sepia` - Adds a sepia tone to your image.

## Example Transformations
- Here are a few examples of what Imagemorpho can do!

### 🎭 ASCII
| **Input** | **Output** |
|-----------|-----------|
| ![Drip Goku Original](ExampleImages/dripgoku.jpg) | ![Drip Goku ASCII](ExampleImages/dripgoku_ascii.jpg) |

### 😃Emoji
| **Input** | **Output** |
|-----------|-----------|
| ![Thragg Original](ExampleImages/thragg.webp) | ![Thragg Emoji](ExampleImages/thragg_emoji.webp) |

### 💥Pop Art
| **Input** | **Output** |
|-----------|-----------|
| ![ohmygah Original](ExampleImages/ohmygah.jpg) | ![ohmygah ASCII](ExampleImages/ohmygah_popart.jpg) |



## Technologies Used

- **Python** – Handles backend logic and image processing
- **Flask** – Manages API requests
- **Mangum** – Makes Flask work with AWS Lambda
- **Discord API** – Lets the bot interact with Discord
- **PIL, OpenCV, NumPy** – Used for image transformations
- **AWS Lambda** – Runs the bot in the cloud without needing servers
- **Docker** – Packages the app into containers
- **TypeScript** – Used in some bot scripts
- **AWS CDK** – Deploys infrastructure with code
- **YAML** – Defines the bot commands

## Contributing

Everyone is welcome to contribute. If you want to improve Imagemorpho or add new transformations, you can submit a pull request.

## License

This project is licensed under the MIT License.

