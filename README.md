# Nanostructure Image Analysis Tool

## Overview

The **Nanostructure Image Analysis Tool** is a Streamlit application designed to analyze microscope images of nanostructures. The tool detects and measures particles in the images, allowing users to remove unwanted areas, perform thresholding, and generate particle statistics.

[Link to the App](https://semanalysisapp-7kuah4ckxxnq5seyqwefcs.streamlit.app/)

## Features

- **Image Upload**: Users can upload microscope images in various formats (PNG, JPG, JPEG, BMP, TIF).
- **White Block Detection**: Automatically detects a white block in the uploaded image and provides its pixel coordinates and dimensions.
- **Calibration**: Users can input the scale bar length in pixels and its equivalent in micrometers to convert pixel measurements to micrometers.
- **Particle Segmentation**: The application allows users to threshold the image to segment particles and measure their properties.
- **Statistical Analysis**: Calculates and displays statistics for the detected particles, including area, perimeter, and equivalent diameter.
- **Outlier Removal**: Users can set a threshold to filter out larger outlier particles.
- **Size Distribution Visualization**: Displays histograms of particle sizes and allows users to customize the x-axis scale.
- **Overlay Visualization**: Provides a visual overlay of segmented particles on the original image.
- **Downloadable Results**: Users can download measurement results as CSV files.

## Requirements

To run this application, ensure you have the following Python packages installed:

- `streamlit`
- `numpy`
- `pandas`
- `opencv-python`
- `scikit-image`
- `matplotlib`

You can install the required packages using pip:

```
pip install streamlit numpy pandas opencv-python scikit-image matplotlib
```

## How to Run the Application

1. Clone this repository to your local machine:
    
    ```
    git clone https://github.com/gadoseb/SEM_Analysis_Streamlit.git
    cd SEM_Analysis_Streamlit   
    ```
    
2. Run the Streamlit application:
    
    ```
    streamlit run app.py
    ```
    
3. Open your web browser and navigate to the provided local URL (usually `http://localhost:8501`).

## Usage Instructions

1. **Upload Image**: Click on the file uploader to choose an image file.
2. **Detect White Block**: The application will automatically analyze the uploaded image for a white block and display its dimensions.
3. **Calibration**: Enter the scale bar length in pixels and the corresponding length in micrometers.
4. **Remove Instrument Information**: Specify the area of the image to be removed.
5. **Thresholding**: Adjust the slider to set the threshold for particle segmentation.
6. **Particle Measurements**: Review the measurements of the detected particles in the displayed table.
7. **Outlier Removal**: Set the outlier threshold to filter larger particles.
8. **Visualizations**: View histograms and overlay images for further analysis.
9. **Download Results**: Use the download buttons to export measurement data as CSV files.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.