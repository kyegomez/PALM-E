# Agora
This implementation of PALM-E is brought to you by Agora, we're a collective of Creators!

[Join us and unleash your creator spirit](https://apac.ai/Agora)

# PALM-E: A Revolutionary Multi-Modal AI Model

[PaLM-E: An Embodied Multimodal Language Model](https://arxiv.org/pdf/2303.03378v1.pdf)


PALM-E is an innovative multi-modal AI model that combines the power of pre-trained language models with continuous observation encoders, such as Vision Transformers (ViT). 

By leveraging the strengths of both text and visual data, PALM-E has the potential to transform the world economy and help people in numerous ways.

How PALM-E Works
PALM-E is built upon the following key components:

A pre-trained Language Model (PaLM) as the base model.

An encoder for continuous observations (e.g., Vision Transformer (ViT)).

A projector to map the encoder output to the language embedding space.

The model processes both text and continuous observations, such as images, by encoding the observations using the chosen encoder and interleaving the encoded observations with text tokens to form multi-modal sentences. This allows PALM-E to understand and generate context-aware responses based on both textual and visual information.


# Get Started

Clone the repository and install the required packages.

```
git clone https://github.com/kyegomez/PALM-E.git
cd palm_e
pip3 install -r requirements.txt
python3 train_distributed.py
```


# How PALM-E Can Help People and Transform the World Economy
PALM-E's ability to process and understand multi-modal data opens up a world of possibilities in various domains, including:

E-commerce: Enhance product recommendations by understanding both textual descriptions and visual features of products.
Healthcare: Improve diagnostics by analyzing medical images and textual patient records simultaneously.
Education: Create personalized learning experiences by understanding students' textual inputs and visual cues.
Smart Cities: Optimize urban planning and resource allocation by analyzing satellite imagery and textual data from various sources.
These are just a few examples of how PALM-E can revolutionize industries and improve people's lives.

# Contribute to PALM-E and Make the World a Better Place
We invite you to join us in our mission to make the world a better place through the power of multi-modal AI. By contributing to the PALM-E project, you can help advance the state of the art in AI and unlock new possibilities for people around the globe.

To get started, please follow these steps:

Fork the PALM-E GitHub repository.
Clone the forked repository to your local machine.
Install the required dependencies.
Explore the code and identify areas where you can contribute.
Create a new branch for your changes.
Commit your changes and push them to your forked repository.
Create a pull request to submit your changes for review.
We welcome contributions in the form of bug fixes, performance improvements, new features, and documentation updates. Together, we can shape the future of AI and create a better world for everyone.

Thank you for your interest in PALM-E, and we look forward to collaborating with you!

# TODO:

* Test to see if the decoder is configured correctly if not we need to pass in embedded tokens, embed positions, output projections

* Recreate the training strategy from the paper where For each example in the dataset: a. Extract text and continuous observations. b. Tokenize text and continuous observations using the tokenizer. c. Encode continuous observations using the chosen encoder. d. Interleave encoded observations with text tokens to form multi-modal sentences. e. Compute loss using cross-entropy for non-prefix tokens. f. Update model parameters.
 

* Train, on the same datasets used in the paper