# FLOORVERSE: A Universe of Floor Plan Possibilities

FLOORVERSE is an innovative platform that leverages cutting-edge AI to automate and optimize the generation of residential floorplans. By utilizing a *Conditional Variational Autoencoder (CVAE)* model, meticulously trained on the extensive **RPlan dataset** (comprising over 80,000 images), FLOORVERSE produces functionally coherent and aesthetically pleasing layouts based on user-defined constraints. The system seamlessly integrates a Flask-based backend with an interactive frontend, providing users with real-time, personalized design experiences.

## Architecture

At the heart of FLOORVERSE is a **Conditional Variational Autoencoder (CVAE)**, designed to process visual data from floorplan images in conjunction with user-defined design parameters. The architecture is structured into the following key components:

1.  **Encoder**:  Utilizes convolutional layers to efficiently extract spatial features directly from grayscale floorplan images.
2.  **Reparameterization Trick**:  Employs this technique to sample latent vectors in a differentiable manner, crucial for effective backpropagation and model training.
3.  **Decoder**:  Reconstructs existing floorplans or generates entirely new ones based on the learned latent representations and the specific user conditions provided.
4.  **Training & Optimization**:  The model is trained using a composite loss function that combines binary cross-entropy for image reconstruction and KL divergence to ensure a well-structured latent space, leading to robust and reliable performance.

## Workflow

The FLOORVERSE workflow follows a streamlined process:

1.  Input Preprocessing: Raw floorplan images undergo preprocessing steps such as resizing and normalization to prepare them for model input.
2.  Feature Encoding: The Encoder network processes the preprocessed visual data, while condition vectors representing user inputs are processed through a multi-layer perceptron to capture relevant design constraints.
3.  Latent Space Representation: Fully connected layers are used to generate a compressed latent space representation that encapsulates the essence of the input floorplan and user conditions.
4.  Floorplan Generation: The Decoder network takes the latent space representation and user constraints to reconstruct or generate new floorplans that are tailored to the specified requirements.

## Evaluation and Metrics

The performance of FLOORVERSE has been rigorously evaluated through a combination of quantitative metrics and qualitative analyses to ensure both accuracy and design quality.

### Quantitative Metrics

*   *Reconstruction Loss*: 8571.3582 - Measures how well the model reconstructs input floorplans, indicating the fidelity of the encoder-decoder process.
*   *Total Loss*: 8654.9813 - Represents the overall loss during training, combining reconstruction loss and KL divergence to optimize model performance.
*   *KL Divergence*: 83.6231 - Quantifies the divergence between the learned latent space distribution and a standard normal distribution, ensuring a well-organized and continuous latent space for generation.

### Qualitative Analyses

*   *Latent Space Traversals*: Demonstrated the ability to smoothly interpolate between diverse design archetypes -  Showcases the model's capability to generate a spectrum of floorplan styles and configurations by navigating the learned latent space.
*   *Principal Component Analysis (PCA)*: Highlighted spatial coherence in generated layouts -  Visualizes the latent space and confirms that generated layouts maintain spatial relationships and structural integrity, resulting in realistic and well-structured floorplans.



## Usage

### Clone the repository
```bash
git clone https://github.com/your-repo/FLOORVERSE.git
cd FLOORVERSE
```
### Prerequisites
```bash
pip install -r requirements.txt
```
### Start the Flask server
```bash
python app.py
```


### Documentation
Comprehensive documentation for FLOORVERSE is available `documentation/` directory to help you understand the project in detail, from setup to advanced usage and development.

## Contributors

This project was a collaborative effort by a team of dedicated individuals, including:

- [Snigdha Pandey](https://github.com/snigdhap2301)
- [Ansh Prakash](https://github.com/anshprakash6397)
- [Esther George Sam](https://github.com/esthersam07)
- [Munish Thakur](https://github.com/menotthakur)
- Naveen Singh



## Acknowledgments
FLOORVERSE is built upon the foundation of extensive research in AI-driven architectural design. We gratefully acknowledge the insights and advancements from prior works in generative adversarial networks, graph neural networks, and diffusion-based approaches within the field.
We extend our sincere gratitude to the creators of the RPlan dataset for providing a robust and invaluable resource that enabled the training of our model.
