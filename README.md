<h1>Handwritten Digit Recognition with CNN</h1>
This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset, achieving approximately 98.7% accuracy on the test set. It demonstrates key machine learning concepts, including model design, data augmentation, visualization, and deployment. Key features include:
<ul>
  <li>**Data Augmentation:** Enhances model generalization using rotation, zoom, and shift transformations.</li>
  <li>**Visualization:** Provides feature maps and a confusion matrix to interpret model behavior.</li>
  <li>**Deployment:** A Streamlit web app allows users to upload images of handwritten digits for real-time predictions.</li>
</ul>

This project is ideal for showcasing skills in deep learning, data visualization, and web deployment, making it a strong addition to a resume.
Table of Contents

<h2>Project Overview</h2>
<ol>
  <li>Installation</li>
  <li>Usage</li>
  <li>Model Architecture</li>
  <li>Results</li>
  <li>Deployment</li>
  <li>License</li>
  <li>Contact</li>
</ol>








<h2>Project Overview</h2>
The project uses the MNIST dataset, which consists of 60,000 training images and 10,000 testing images of handwritten digits (0-9), each a 28x28 grayscale image. The dataset is loaded via Keras, requiring no external downloads. The CNN is built using Keras (part of TensorFlow), trained with data augmentation to improve robustness, and evaluated with visualizations for interpretability. The trained model is deployed as a Streamlit web app, allowing users to upload images and receive digit predictions.
Technologies used:
<ul>
  <li>Python</li>
  <li>TensorFlow/Keras for model building and training</li>
  <li>NumPy for data preprocessing</li>
  <li>Scikit-learn, Seaborn, and Matplotlib for evaluation and visualization</li>
  <li>Streamlit and Pillow for web deployment and image processing
</li>
</ul>


<h2>Installation</h2>
To run this project, install the following Python packages:
<table border="1" cellpadding="8" cellspacing="0">
  <thead>
    <tr>
      <th>Package</th>
      <th>Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>tensorflow</td>
      <td>Build and train the CNN (includes Keras)</td>
    </tr>
    <tr>
      <td>numpy</td>
      <td>Data preprocessing and numerical operations</td>
    </tr>
    <tr>
      <td>scikit-learn</td>
      <td>Generate confusion matrix for evaluation</td>
    </tr>
    <tr>
      <td>seaborn</td>
      <td>Visualize confusion matrix</td>
    </tr>
    <tr>
      <td>matplotlib</td>
      <td>Plot feature maps and other visualizations</td>
    </tr>
    <tr>
      <td>streamlit</td>
      <td>Deploy the web app</td>
    </tr>
    <tr>
      <td>pillow</td>
      <td>Process images in the Streamlit app</td>
    </tr>
  </tbody>
</table>






**Install all dependencies using pip:**
pip install tensorflow numpy scikit-learn seaborn matplotlib streamlit pillow

Alternatively, create a requirements.txt file with the following content and install using:
pip install -r requirements.txt

**requirements.txt:**
tensorflow
numpy
scikit-learn
seaborn
matplotlib
streamlit
pillow

Note: Use Python 3.8 or later for compatibility. A virtual environment is recommended to avoid conflicts.
Usage
The project includes two main scripts: train.py for training and evaluation, and app.py for the Streamlit app.

  <h2>Train the Model</h2>
  <p><strong>Run:</strong> <code>python train.py</code></p>
  <p>This script:</p>
  <ul>
    <li>Loads and preprocesses the MNIST dataset.</li>
    <li>Builds and trains the CNN model.</li>
    <li>Evaluates the model on the test set.</li>
    <li>Generates visualizations (confusion matrix and feature maps) saved in the <code>images/</code> directory.</li>
    <li>Saves the trained model as <code>mnist_cnn.h5</code>.</li>
  </ul>

  <h2>Run the Streamlit App</h2>
  <ul>
    <li>Ensure the model is trained and saved as <code>mnist_cnn.h5</code>.</li>
    <li><strong>Run:</strong> <code>streamlit run app.py</code></li>
  </ul>
  <p>This starts a web server. Open the provided URL in your browser to upload images and see predictions.</p>

  <h2>Model Architecture</h2>
  <p>The CNN is designed to achieve high accuracy on the MNIST dataset. Its architecture includes:</p>

  <table border="1" cellpadding="8" cellspacing="0">
    <thead>
      <tr>
        <th>Layer Type</th>
        <th>Parameters</th>
        <th>Output Shape</th>
        <th>Purpose</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Conv2D</td>
        <td>32 filters, 3x3, ReLU</td>
        <td>(26, 26, 32)</td>
        <td>Detect basic features (e.g., edges)</td>
      </tr>
      <tr>
        <td>MaxPooling2D</td>
        <td>2x2 pool</td>
        <td>(13, 13, 32)</td>
        <td>Reduce spatial dimensions</td>
      </tr>
      <tr>
        <td>Conv2D</td>
        <td>64 filters, 3x3, ReLU</td>
        <td>(11, 11, 64)</td>
        <td>Detect complex features</td>
      </tr>
      <tr>
        <td>MaxPooling2D</td>
        <td>2x2 pool</td>
        <td>(5, 5, 64)</td>
        <td>Further reduce dimensions</td>
      </tr>
      <tr>
        <td>Flatten</td>
        <td>-</td>
        <td>(1600,)</td>
        <td>Prepare for dense layers</td>
      </tr>
      <tr>
        <td>Dense</td>
        <td>128 units, ReLU</td>
        <td>(128,)</td>
        <td>Learn high-level patterns</td>
      </tr>
      <tr>
        <td>Dropout</td>
        <td>50% dropout</td>
        <td>(128,)</td>
        <td>Prevent overfitting</td>
      </tr>
      <tr>
        <td>Dense</td>
        <td>10 units, softmax</td>
        <td>(10,)</td>
        <td>Output probabilities for 10 classes</td>
      </tr>
    </tbody>
  </table>

  <h2>Data Augmentation</h2>
  <p>Applied using Keras' <code>ImageDataGenerator</code> with:</p>
  <ul>
    <li><code>rotation_range=10</code></li>
    <li><code>zoom_range=0.1</code></li>
    <li><code>width_shift_range=0.1</code></li>
    <li><code>height_shift_range=0.1</code></li>
  </ul>
  <p>These techniques improve the model’s ability to generalize to varied inputs.</p>

  <h2>Results</h2>
  <p>The model achieves approximately <strong>98.7% accuracy</strong> on the MNIST test set, though slight variations may occur due to training randomness.</p>
  <p>Visualizations include:</p>
  <ul>
    <li><strong>Confusion Matrix:</strong> Shows classification performance across digit classes, saved as <code>images/confusion_matrix.png</code>.</li>
    <li><strong>Feature Maps:</strong> Illustrates features learned by convolutional layers, saved as <code>images/feature_maps_layer1.png</code> and <code>images/feature_maps_layer2.png</code>.</li>
  </ul>
  <p>Run <code>train.py</code> to generate these visualizations and view them in the <code>images/</code> directory.</p>

  <h2>Deployment</h2>
  <p>The model is deployed as a Streamlit web app, allowing users to upload images (JPEG or PNG) of handwritten digits for prediction. The app:</p>
  <ul>
    <li>Preprocesses the image (resizes to 28x28, converts to grayscale)</li>
    <li>Uses the trained model to predict the digit</li>
  </ul>
  <p><strong>To run locally:</strong></p>
  <ul>
    <li>Ensure <code>mnist_cnn.h5</code> exists.</li>
    <li>Execute: <code>streamlit run app.py</code></li>
  </ul>
  <p>Access the app in your browser to upload images and view predictions.</p>

  <h2>License</h2>
  <p>This project is licensed under the MIT License. See the <code>LICENSE</code> file for details.</p>

