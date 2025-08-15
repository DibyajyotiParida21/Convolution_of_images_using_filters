import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from scipy.signal import convolve2d
from scipy.ndimage import convolve

st.title('Convolution of an image')

# Optional: upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

#st.image(img, caption="Orignal image", use_container_width=True)

# Dictionary of filters and their kernels
filter_kernels = {
    "Edge Detection Filters - Sobel Filter (X)": np.array([[-1, 0, 1],
                                                           [-2, 0, 2],
                                                           [-1, 0, 1]]),
    "Edge Detection Filters - Prewitt Filter": np.array([[-1, 0, 1],
                                                         [-1, 0, 1],
                                                         [-1, 0, 1]]),
    "Edge Detection Filters - Laplacian Filter": np.array([[0, -1, 0],
                                                           [-1, 4, -1],
                                                           [0, -1, 0]]),
    "Smoothening/Blurring - Averaging Filter": (1/9) * np.ones((3, 3)),
    "Smoothening/Blurring - Gaussian Blur": (1/16) * np.array([[1, 2, 1],
                                                               [2, 4, 2],
                                                               [1, 2, 1]]),
    "Sharpening Filters": np.array([[0, -1, 0],
                                    [-1, 5, -1],
                                    [0, -1, 0]]),
    "3D effect - Emboss Kernel": np.array([[-2, -1, 0],
                                           [-1, 1, 1],
                                           [0, 1, 2]])
}

# Store last applied filter
if "applied_filter" not in st.session_state:
    st.session_state.applied_filter = None

# Filter picker + Apply button
selected_filter = st.selectbox("Choose an edge filter", list(filter_kernels.keys()))
if st.button("Apply"):
    st.session_state.applied_filter = selected_filter

# Render only after Apply
if st.session_state.applied_filter is not None:
    kernel = filter_kernels[st.session_state.applied_filter]

    st.subheader("Selected Filter Kernel (3×3 Matrix):")
    st.write(kernel)

    # Build matrix plot with limited grey range
    fig, ax = plt.subplots(figsize=(1, 1), dpi=320)

    # Limit vmin/vmax so colors stay mid-grey (0.3 to 0.8 of full scale)
    im = ax.matshow(kernel, cmap="Purples" , vmin=np.min(kernel) * 0.3, vmax=np.max(kernel) * 0.7)

    # Add text values
    for (i, j), val in np.ndenumerate(kernel):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=6)

    ax.axis("off")

    # Save to bytes
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # Center image
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image(buf, width=300, caption=st.session_state.applied_filter)
    
    left, right = st.columns([4, 1], gap="small")
    with left:
        st.markdown(
        "<h3 style='margin-top: 0.4em;'>Click here to generate new modified image</h3>",
        unsafe_allow_html=True
    )

    with right:
        st.markdown("<div style='margin-top: 1.4em;'>", unsafe_allow_html=True)
        x = st.button("Click me", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if x:
        # now we shall perform convolution of image data with the filter chosen

        img = Image.open(uploaded_file)
        img.save("compressed.jpg", quality=30, optimize=True)

        # Gray scale new image
        # converting an RGB/grayscale image to grayscale 
        gray_img = img.convert("L")
        img_array = np.array(gray_img)
        # Perform convolution
        convolved = convolve2d(img_array, kernel, mode='same', boundary='fill', fillvalue=0)
        # Normalize to 0–255 for display
        convolved = np.clip(convolved, 0, 255).astype(np.uint8)

        # Convert back to Image
        result_img = Image.fromarray(convolved)
        #result_img.show()

        # RGB new image
        img_array_1 = np.array(img)  # Shape: (H, W, 3)

        # Apply convolution to each channel automatically
        convolved_rgb = np.zeros_like(img_array_1)
        for c in range(3):
            convolved_rgb[:, :, c] = convolve(img_array_1[:, :, c], kernel, mode='constant', cval=0.0)

        result_img_1 = Image.fromarray(np.clip(convolved_rgb, 0, 255).astype(np.uint8))
        #st.image(result_img_1, caption="Convolved RGB Image", use_container_width=True)
        #result_img_1.show()

        gray_side, RGB_side = st.columns([1,1])
        with gray_side:
            st.image(result_img, caption="Gray scale modified image", use_container_width=True)
        with RGB_side:
            st.image(result_img_1, caption="RGB modified image", use_container_width=True)

        st.write("In RGB format")

        img_array_2 = np.array(result_img_1)

        # Extract each channel
        r_channel = img_array_2[:, :, 0]
        g_channel = img_array_2[:, :, 1]
        b_channel = img_array_2[:, :, 2]

        # Convert single-channel grayscale to colored channel
        r_img = np.stack([r_channel, np.zeros_like(r_channel), np.zeros_like(r_channel)], axis=2)
        g_img = np.stack([np.zeros_like(g_channel), g_channel, np.zeros_like(g_channel)], axis=2)
        b_img = np.stack([np.zeros_like(b_channel), np.zeros_like(b_channel), b_channel], axis=2)

        R, G, B = st.columns(3)
        with R:
            st.image(r_img, caption="Red Channel", use_container_width=True)
        with G:
            st.image(g_img, caption="Green Channel", use_container_width=True)
        with B:
            st.image(b_img, caption="Blue Channel", use_container_width=True)