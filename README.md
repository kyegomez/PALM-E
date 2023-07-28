# Agora
This implementation of PALM-E is brought to you by Agora, we're a collective of Creators!

[Join us and unleash your creator spirit](https://apac.ai/Agora)

# PALM-E: A Revolutionary Multi-Modal AI Model

[PaLM-E: An Embodied Multimodal Language Model](https://arxiv.org/pdf/2303.03378v1.pdf)

PALM-E is an innovative multi-modal AI model that combines the power of pre-trained language models with continuous observation encoders, such as Vision Transformers (ViT). 

## Value Proposition

PALM-E creates value through:

- Maximize Dream Outcome: Provides a solution to integrate both visual and textual data for problem-solving.
- Maximize Perceived Likelihood of Success: Incorporates proven technologies like pre-trained Language Models and Vision Transformers.
- Minimize Time to Success: Optimized with fast-processing encoders and projectors.
- Minimize Effort & Sacrifice: Simplifies complex tasks of multi-modal sentence formation.

## Model Architecture

PALM-E is built upon the following key components:

- A pre-trained Language Model (PaLM) as the base model.
- An encoder for continuous observations (e.g., Vision Transformer (ViT)).
- A projector to map the encoder output to the language embedding space.

PALM-E processes both text and continuous observations, such as images, and forms multi-modal sentences by interleaving the encoded observations with text tokens. This allows it to generate context-aware responses based on both textual and visual information.

## Installation

Clone the repository and install the required packages.

```sh
git clone https://github.com/kyegomez/PALM-E.git
cd palm_e
pip install -r requirements.txt
```

You can also install PALM-E directly using pip:

```sh
pip install git+https://github.com/kyegomez/PALM-E.git
```

Then, run the training script:

```sh
python3 train_distributed.py
```

## Commercial Use Cases

PALM-E's ability to process and understand multi-modal data opens up a world of possibilities in various domains, including:

- E-commerce: Enhance product recommendations by understanding both textual descriptions and visual features of products.
- Healthcare: Improve diagnostics by analyzing medical images and textual patient records simultaneously.
- Education: Create personalized learning experiences by understanding students' textual inputs and visual cues.
- Smart Cities: Optimize urban planning and resource allocation by analyzing satellite imagery and textual data from various sources.

These are just a few examples of how PALM-E can revolutionize industries and improve people's lives.

## Contribute to PALM-E and Make the World a Better Place

We invite you to join us in our mission to make the world a better place through the power of multi-modal AI. Here are the steps to start contributing:

1. Fork the PALM-E GitHub repository.
2. Clone the forked repository to your local machine.
3. Install the required dependencies.
4. Explore the code and identify areas where you can contribute.
5. Create a new branch for your changes.
6. Commit your changes and push them to your forked repository.
7. Create a pull request to submit your changes for review.

We welcome contributions in any form of bug fixes, performance improvements, new features, and documentation updates. Together, we can shape the future of AI and create a better world for everyone.

Thank you for your interest in PALM-E, and we look forward to collaborating with you!

# Future Work:

* Verify the correct configuration of the decoder. If necessary, pass in embedded tokens, embed positions, output projections.
* Recreate the training strategy from the paper:
    - For each example in the dataset:
        - Extract text and continuous observations.
        - Tokenize text and continuous observations using the tokenizer.
        - Encode continuous observations using the chosen encoder.
        - Interleave encoded observations with text tokens to form multi-modal sentences.
        - Compute loss using cross-entropy for non-prefix tokens.
        - Update model parameters.
* Train the model on the same datasets used in the paper.

# Examples and Documentation

Please refer to the "examples" and "docs" directories for more detailed examples and comprehensive documentation on using PALM-E. Feel free to raise an issue or contact us directly for further inquiries.