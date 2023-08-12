# 🌴 PALM-E: A Multi-Modal AI Model 

Palm-E can Process both text and images in harmony.


[![GitHub issues](https://img.shields.io/github/issues/kyegomez/PALM-E)](https://github.com/kyegomez/PALM-E/issues) 
[![GitHub forks](https://img.shields.io/github/forks/kyegomez/PALM-E)](https://github.com/kyegomez/PALM-E/network) 
[![GitHub stars](https://img.shields.io/github/stars/kyegomez/PALM-E)](https://github.com/kyegomez/PALM-E/stargazers) [![GitHub license](https://img.shields.io/github/license/kyegomez/PALM-E)](https://github.com/kyegomez/PALM-E/blob/master/LICENSE)
[![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/PALM-E)](https://twitter.com/intent/tweet?text=Excited%20to%20introduce%20PALM-E,%20the%20all-new%20robotics%20model%20with%20the%20potential%20to%20revolutionize%20automation.%20Join%20us%20on%20this%20journey%20towards%20a%20smarter%20future.%20%23RT1%20%23Robotics&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FPALM-E)
[![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FPALM-E)
[![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FPALM-E&title=Introducing%20PALM-E%2C%20the%20All-New%20Robotics%20Model&summary=PALM-E%20is%20the%20next-generation%20robotics%20model%20that%20promises%20to%20transform%20industries%20with%20its%20intelligence%20and%20efficiency.%20Join%20us%20to%20be%20a%20part%20of%20this%20revolutionary%20journey%20%23RT1%20%23Robotics&source=)
![Discord](https://img.shields.io/discord/999382051935506503)
[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FPALM-E&title=Exciting%20Times%20Ahead%20with%20PALM-E%2C%20the%20All-New%20Robotics%20Model%20%23RT1%20%23Robotics) [![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FPALM-E&t=Exciting%20Times%20Ahead%20with%20PALM-E%2C%20the%20All-New%20Robotics%20Model%20%23RT1%20%23Robotics)
[![Share on Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FPALM-E&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=PALM-E%2C%20the%20Revolutionary%20Robotics%20Model%20that%20will%20Change%20the%20Way%20We%20Work%20%23RT1%20%23Robotics)
[![Share on WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=I%20just%20discovered%20PALM-E,%20the%20all-new%20robotics%20model%20that%20promises%20to%20revolutionize%20automation.%20Join%20me%20on%20this%20exciting%20journey%20towards%20a%20smarter%20future.%20%23RT1%20%23Robotics%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2FPALM-E)



## 🚀 Quick Start

### Installation 📦

```sh
pip install palme
```

### Usage 🎨

```python
import torch
from palme.model import PalmE

# Sample text token tensor
text_tokens = torch.randint(0, 32002, (1, 50), dtype=torch.long)

# Sample image tensor
images = torch.randn(1, 3, 224, 224)

model = PalmE()

output = model.forward(text_tokens=text_tokens, images=images)

print(f"Output: {output}")
```
---

## Features

- 🌈 **Maximize Dream Outcome**: A key to integrating both visual and textual data for problem-solving.
- 🎯 **Maximize Perceived Likelihood of Success**: Built with the giants - pre-trained Language Models and Vision Transformers.
- ⏩ **Minimize Time to Success**: Speedy Gonzales thanks to fast-processing encoders and projectors.
- 🍰 **Minimize Effort & Sacrifice**: We simplify the daunting tasks of multi-modal sentence formation for you!

## 🔍 System Architecture

- 📜 A majestic pre-trained Language Model.
- 🖼️ An elegant encoder for continuous observations (like Vision Transformer).
- 🔦 A projector that's perfect to map the encoder outputs to the language embedding realms.

Our approach processes both text and continuous observations, intertwining the encoded observations with text tokens for multi-modal sentences. Hence, context-aware responses based on textual and visual info!

## Real-World Use-Cases🌍 

- 🛍️ **E-commerce**: Up your recommendation game with text + visual understanding.
- 🌡️ **Healthcare**: A 360° patient view with image & text data.
- 📚 **Education**: Tailored learning experiences.
- 🏙️ **Smart Cities**: Plan, optimize, and beautify urban spaces with satellite imagery and textual data insights.

... and that's just scratching the surface! 🚀

## Contribute || Be Part of the PALM-E Adventure 🤝

Your brilliance is needed! Join us, and together, let's make PALM-E even more awe-inspiring:

1. **Get Your Copy**: Fork the PALM-E repo.
2. **Make It Local**: Clone your fork.
3. **Prep Your Tools**: Install the necessities.
4. **Discover & Innovate**: Dive into the code.
5. **Craft Your Magic**: Branch and code away.
6. **Show & Tell**: Push your changes and craft a pull request.

🐞 Fixes, 🎨 enhancements, 📝 docs, or 💡 ideas – all are welcome! Let's shape the future of AI, hand in hand.

## Roadmap

- 🕵️ Verify decoder configurations.
- 🚂 Recreate the training strategy detailed in the paper.
- 🌐 Train on the datasets used in the paper.

## 📘 Documentation
* Documentation will come soon