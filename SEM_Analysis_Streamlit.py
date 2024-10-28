import streamlit as st
import numpy as np
import pandas as pd
import cv2
from skimage import measure
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt

# Function to detect the white block
def detect_white_block(image):
    # Convert to binary image
    _, binary_image = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If any contours are found, return the largest one
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, w, h)
    return None

# Streamlit app
st.title("Nanostructure Image Analysis Tool")
st.write("Upload a microscope image of nanostructures for analysis.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "bmp", "tif"])

if uploaded_file is not None:
    # Save the uploaded image for OpenCV processing
    image_path = "/tmp/uploaded_image.png"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the image for analysis
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Display the image
    st.image(image_path, caption="Uploaded Image", use_column_width=True)

    # Detect the white block
    block_info = detect_white_block(image)
    
    if block_info is not None:
        x, y, w, h = block_info
        st.write(f"Detected white block at: x={x}, y={y}, width={w}, height={h} pixels")
        
        # Draw a rectangle around the detected block for visualization
        detected_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(detected_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add axes to show pixel coordinates
        fig, ax = plt.subplots()
        ax.imshow(detected_image)
        ax.set_title("Detected White Block with Pixel Coordinates")
        ax.set_xlabel("Pixel X Coordinate")
        ax.set_ylabel("Pixel Y Coordinate")
        ax.axis('on')  # Turn on the axis to show the pixel scale
        st.pyplot(fig)
        
        # Calibration section
        scale_bar_length_pixels = st.number_input("Enter scale bar length in pixels", min_value=1, step=1, value=w)
        scale_bar_um = st.number_input("Scale bar length in micrometers (µm)", min_value=0.01, step=0.01)

        if scale_bar_length_pixels > 0 and scale_bar_um > 0:
            pixel_to_micron_ratio = scale_bar_um / scale_bar_length_pixels
            st.write(f"Pixel-to-micron ratio: {pixel_to_micron_ratio:.4f} µm/px")

            # Remove the area with instrument information
            st.subheader("Remove Instrument Information Area")
            area_x = st.number_input("Enter x-coordinate of the area to remove", min_value=0, step=1, value=0)
            area_y = st.number_input("Enter y-coordinate of the area to remove", min_value=0, step=1, value=0)
            area_width = st.number_input("Enter width of the area to remove", min_value=1, step=1, value=image.shape[1])
            area_height = st.number_input("Enter height of the area to remove", min_value=1, step=1, value=50)  # Adjust height as needed
            
            # Create a mask to remove the specified area
            mask = np.ones(image.shape, dtype=np.uint8) * 255  # Create a white mask
            mask[area_y:area_y + area_height, area_x:area_x + area_width] = 0  # Set the specified area to black
            masked_image = cv2.bitwise_and(image, image, mask=mask)

            # Thresholding and segmentation
            st.subheader("Thresholding and Particle Segmentation")
            threshold = st.slider("Threshold value", 0, 255, 128)
            _, binary_image = cv2.threshold(masked_image, threshold, 255, cv2.THRESH_BINARY)
            st.image(binary_image, caption="Binary Image (Thresholded)", use_column_width=True)

            # Connected component labeling
            label_img = label(binary_image)
            props = regionprops(label_img)

            # Measurements and Analysis
            st.subheader("Particle Measurements")
            data = []
            for i, prop in enumerate(props):
                area = prop.area * pixel_to_micron_ratio**2
                perimeter = prop.perimeter * pixel_to_micron_ratio
                equivalent_diameter = prop.equivalent_diameter * pixel_to_micron_ratio
                data.append({
                    "Particle": i + 1,
                    "Area (µm^2)": area,
                    "Perimeter (µm)": perimeter,
                    "Equivalent Diameter (µm)": equivalent_diameter
                })
            
            df = pd.DataFrame(data)
            st.write(df)

            # Option to download the results
            csv = df.to_csv(index=False)
            st.download_button(label="Download measurements as CSV", data=csv, mime="text/csv")

            # Outlier removal section
            st.subheader("Remove Larger Outliers")
            outlier_threshold = st.number_input("Set outlier threshold for equivalent diameter (µm)", min_value=0.0, step=0.1)

            filtered_df = df  # Initialize filtered_df with the original DataFrame
            if st.button("Remove Outliers"):
                filtered_df = df[df["Equivalent Diameter (µm)"] <= outlier_threshold]
                st.write(f"Filtered Measurements (Outliers removed above {outlier_threshold} µm):")
                st.write(filtered_df)

                # Option to download the filtered results
                filtered_csv = filtered_df.to_csv(index=False)
                st.download_button(label="Download filtered measurements as CSV", data=filtered_csv, mime="text/csv")

            # Plot size distribution
            st.subheader("Size Distribution")
            fig, ax = plt.subplots()
            ax.hist(filtered_df["Equivalent Diameter (µm)"], bins=10, color='skyblue', edgecolor='black')
            ax.set_title("Particle Size Distribution")
            ax.set_xlabel("Equivalent Diameter (µm)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

            # Second plot with user-defined scale
            st.subheader("Size Distribution with Custom Scale in Nanometers")
            x_scale_min = st.number_input("Set minimum x-axis scale (nm)", value=0, step=1)
            x_scale_max = st.number_input("Set maximum x-axis scale (nm)", value=int(filtered_df["Equivalent Diameter (µm)"].max() * 1000), step=1)

            if x_scale_min < x_scale_max:
                fig2, ax2 = plt.subplots()
                ax2.hist(filtered_df[(filtered_df["Equivalent Diameter (µm)"] * 1000 >= x_scale_min) & (filtered_df["Equivalent Diameter (µm)"] * 1000 <= x_scale_max)]["Equivalent Diameter (µm)"] * 1000, 
                             bins=10, color='skyblue', edgecolor='black')
                ax2.set_title("Particle Size Distribution with Custom Scale (nm)")
                ax2.set_xlabel("Equivalent Diameter (nm)")
                ax2.set_ylabel("Frequency")
                ax2.set_xlim(x_scale_min, x_scale_max)
                st.pyplot(fig2)

            # Statistical analysis
            if not filtered_df.empty:
                st.subheader("Particle Measurements Statistics")
                mean_area = filtered_df["Area (µm^2)"].mean()
                median_area = filtered_df["Area (µm^2)"].median()
                std_area = filtered_df["Area (µm^2)"].std()

                mean_perimeter = filtered_df["Perimeter (µm)"].mean()
                median_perimeter = filtered_df["Perimeter (µm)"].median()
                std_perimeter = filtered_df["Perimeter (µm)"].std()

                mean_equiv_diameter = filtered_df["Equivalent Diameter (µm)"].mean()
                median_equiv_diameter = filtered_df["Equivalent Diameter (µm)"].median()
                std_equiv_diameter = filtered_df["Equivalent Diameter (µm)"].std()

                # Display statistics
                st.write(f"Mean Area (µm²): {mean_area:.2f}")
                st.write(f"Median Area (µm²): {median_area:.2f}")
                st.write(f"Standard Deviation of Area (µm²): {std_area:.2f}")
                st.write(f"Mean Perimeter (µm): {mean_perimeter:.2f}")
                st.write(f"Median Perimeter (µm): {median_perimeter:.2f}")
                st.write(f"Standard Deviation of Perimeter (µm): {std_perimeter:.2f}")
                st.write(f"Mean Equivalent Diameter (µm): {mean_equiv_diameter:.2f}")
                st.write(f"Median Equivalent Diameter (µm): {median_equiv_diameter:.2f}")
                st.write(f"Standard Deviation of Equivalent Diameter (µm): {std_equiv_diameter:.2f}")

                # Overlay Plot
                st.subheader("Overlay of Segmented Particles on Original Image")
                overlay_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for overlay

                # Create a colored overlay for the segmented particles
                colored_overlay = np.zeros_like(overlay_image)
                for prop in props:
                    coords = prop.coords
                    for coord in coords:
                        colored_overlay[coord[0], coord[1]] = [0, 255, 0]  # Green color for particles

                # Combine original image and overlay
                combined_image = cv2.addWeighted(overlay_image, 0.7, colored_overlay, 0.3, 0)

                # Show overlay image
                st.image(combined_image, caption="Overlay of Segmented Particles", use_column_width=True)
            
            # Add a button to visualize the original binary image and the labeled particles
            if st.button("Show Binary Image with Labels"):
                # Create a labeled image for visualization
                labeled_image = label(binary_image)

                # Create a figure to display the original binary image and the labeled image
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Create a 1x2 subplot

                # Show the original binary image
                ax[0].imshow(binary_image, cmap='gray')
                ax[0].set_title("Original Binary Image")
                ax[0].axis('off')  # Hide axes

                # Show the labeled image
                ax[1].imshow(labeled_image, cmap='nipy_spectral')
                ax[1].set_title("Labeled Particles")
                ax[1].axis('off')  # Hide axes

                st.pyplot(fig)  # Display the figure in Streamlit

    else:
        st.write("No white block detected. Please ensure the reference block is clearly visible.")