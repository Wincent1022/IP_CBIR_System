import os
import cv2
import pickle
import tempfile
import zipfile
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="CBIR Image Retrieval System",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Content-Based Image Retrieval (CBIR) System For Vehicle")
st.write("Upload an image, choose an algorithm, and retrieve visually similar images from the dataset.")


# =========================================================
# Paths
# =========================================================
SAVE_PATH = "saved_features"


# =========================================================
# Load Saved Data
# =========================================================
@st.cache_data
def load_pickle_file(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_all_data():
    color_features = load_pickle_file(os.path.join(SAVE_PATH, "color_features.pkl"))
    glcm_features = load_pickle_file(os.path.join(SAVE_PATH, "glcm_features.pkl"))
    hu_features = load_pickle_file(os.path.join(SAVE_PATH, "hu_features.pkl"))
    orb_descriptors_db = load_pickle_file(os.path.join(SAVE_PATH, "orb_descriptors.pkl"))
    image_paths = load_pickle_file(os.path.join(SAVE_PATH, "image_paths.pkl"))
    image_labels = load_pickle_file(os.path.join(SAVE_PATH, "image_labels.pkl"))

    return (
        color_features,
        glcm_features,
        hu_features,
        orb_descriptors_db,
        image_paths,
        image_labels
    )

# =========================================================
# Utility Function for Cloud Path
# =========================================================
def normalize_cloud_path(path):
    """
    Convert Windows-style stored paths into paths that work on Streamlit Cloud/Linux.
    """
    if path is None:
        return None

    # Convert backslashes to forward slashes first
    path = path.replace("\\", "/")

    # Normalize the path for the current OS
    path = os.path.normpath(path)

    return path

try:
    (
        color_features,
        glcm_features,
        hu_features,
        orb_descriptors_db,
        image_paths,
        image_labels
    ) = load_all_data()

except Exception as e:
    st.error(f"Error loading saved feature files: {e}")
    st.stop()

image_paths = [normalize_cloud_path(p) for p in image_paths]
    
# =========================================================
# Session State Initialization
# =========================================================
if "retrieved_results" not in st.session_state:
    st.session_state.retrieved_results = None

if "filtered_results" not in st.session_state:
    st.session_state.filtered_results = None

if "selected_algorithm" not in st.session_state:
    st.session_state.selected_algorithm = None

if "query_image_path" not in st.session_state:
    st.session_state.query_image_path = None


# =========================================================
# Utility Functions
# =========================================================
def load_rgb_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_to_bytes(image_path):
    with open(image_path, "rb") as f:
        return f.read()

def create_zip_from_results(results):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for idx, result in enumerate(results, start=1):
            file_name = f"{idx}_{os.path.basename(result['image_path'])}"
            zip_file.writestr(file_name, image_to_bytes(result["image_path"]))
    zip_buffer.seek(0)
    return zip_buffer


def plot_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


# =========================================================
# Feature Extraction Functions
# =========================================================
def extract_color_histogram(image_path, bins=(8, 8, 8)):
    image = cv2.imread(image_path)

    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hist = cv2.calcHist(
        [image], [0, 1, 2], None, bins,
        [0, 256, 0, 256, 0, 256]
    )

    hist = cv2.normalize(hist, hist).flatten()
    return hist


def extract_glcm_features(image_path, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    glcm = graycomatrix(
        image,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()

    return np.hstack([contrast, correlation, energy, homogeneity])


def extract_glcm_property_dict(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    glcm = graycomatrix(
        image,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )

    return {
        "Contrast": float(np.mean(graycoprops(glcm, "contrast"))),
        "Correlation": float(np.mean(graycoprops(glcm, "correlation"))),
        "Energy": float(np.mean(graycoprops(glcm, "energy"))),
        "Homogeneity": float(np.mean(graycoprops(glcm, "homogeneity")))
    }


def extract_hu_moments(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments).flatten()

    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu


def get_hu_binary_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh


orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def extract_orb_descriptors(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    keypoints, descriptors = orb.detectAndCompute(image, None)
    return descriptors


def extract_orb_keypoints_and_descriptors(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None

    keypoints, descriptors = orb.detectAndCompute(image, None)
    return image, keypoints, descriptors


def compute_orb_similarity(query_descriptors, train_descriptors):
    if query_descriptors is None or train_descriptors is None:
        return 0

    matches = bf.match(query_descriptors, train_descriptors)

    if len(matches) == 0:
        return 0

    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = [m for m in matches if m.distance < 50]

    return len(good_matches)


# =========================================================
# Retrieval Functions
# =========================================================
def retrieve_by_similarity(query_image_path, feature_type="color", top_k=5):
    if feature_type == "color":
        query_feature = extract_color_histogram(query_image_path)
        database = color_features
    elif feature_type == "glcm":
        query_feature = extract_glcm_features(query_image_path)
        database = glcm_features
    elif feature_type == "hu":
        query_feature = extract_hu_moments(query_image_path)
        database = hu_features
    else:
        return []

    if query_feature is None:
        return []

    similarities = cosine_similarity([query_feature], database)[0]
    sorted_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in sorted_indices:
        results.append({
            "image_path": image_paths[idx],
            "label": image_labels[idx],
            "score": float(similarities[idx])
        })

        if len(results) == top_k:
            break

    return results


def retrieve_by_orb(query_image_path, top_k=5):
    query_desc = extract_orb_descriptors(query_image_path)

    if query_desc is None:
        return []

    results = []

    for i in range(len(image_paths)):
        score = compute_orb_similarity(query_desc, orb_descriptors_db[i])

        results.append({
            "image_path": image_paths[i],
            "label": image_labels[i],
            "score": float(score)
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# =========================================================
# Score Filtering
# =========================================================
def filter_results_by_threshold(results, threshold_percent, algorithm):
    filtered = []

    if algorithm in ["Color Histogram", "GLCM Texture", "Hu Moments"]:
        threshold_value = threshold_percent / 100.0

        for r in results:
            if r["score"] >= threshold_value:
                filtered.append(r)

    elif algorithm == "ORB":
        if len(results) == 0:
            return []

        max_score = max(r["score"] for r in results)

        if max_score == 0:
            return []

        for r in results:
            normalized_score = (r["score"] / max_score) * 100
            if normalized_score >= threshold_percent:
                r["normalized_score"] = normalized_score
                filtered.append(r)

    return filtered


# =========================================================
# Visualization Functions
# =========================================================
def show_color_analysis(query_image_path, result_image_path):
    st.subheader("Color Analysis")

    query_img = load_rgb_image(query_image_path)
    result_img = load_rgb_image(result_image_path)

    if query_img is None or result_img is None:
        st.warning("Unable to load images for color analysis.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title("Query Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(result_img)
    axes[0, 1].set_title("Top Retrieved Image")
    axes[0, 1].axis("off")

    colors = ("r", "g", "b")
    for i, color in enumerate(colors):
        hist = cv2.calcHist([query_img], [i], None, [256], [0, 256])
        axes[1, 0].plot(hist, color=color)
    axes[1, 0].set_title("Query RGB Histogram")
    axes[1, 0].set_xlim([0, 256])

    for i, color in enumerate(colors):
        hist = cv2.calcHist([result_img], [i], None, [256], [0, 256])
        axes[1, 1].plot(hist, color=color)
    axes[1, 1].set_title("Retrieved RGB Histogram")
    axes[1, 1].set_xlim([0, 256])

    st.pyplot(fig)

    query_mean = query_img.reshape(-1, 3).mean(axis=0)
    result_mean = result_img.reshape(-1, 3).mean(axis=0)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Query Average RGB**")
        st.write(f"R: {query_mean[0]:.2f}, G: {query_mean[1]:.2f}, B: {query_mean[2]:.2f}")
    with col2:
        st.markdown("**Retrieved Average RGB**")
        st.write(f"R: {result_mean[0]:.2f}, G: {result_mean[1]:.2f}, B: {result_mean[2]:.2f}")


def show_glcm_analysis(query_image_path, result_image_path):
    st.subheader("Texture Analysis (GLCM)")

    query_stats = extract_glcm_property_dict(query_image_path)
    result_stats = extract_glcm_property_dict(result_image_path)

    if query_stats is None or result_stats is None:
        st.warning("Unable to compute GLCM statistics.")
        return

    # Show contrast separately
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Query Contrast", f"{query_stats['Contrast']:.4f}")
    with col2:
        st.metric("Retrieved Contrast", f"{result_stats['Contrast']:.4f}")

    # Plot only smaller-scale properties
    properties = ["Correlation", "Energy", "Homogeneity"]
    query_values = [query_stats[p] for p in properties]
    result_values = [result_stats[p] for p in properties]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(properties))
    width = 0.35

    ax.bar(x - width / 2, query_values, width, label="Query")
    ax.bar(x + width / 2, result_values, width, label="Retrieved")
    ax.set_xticks(x)
    ax.set_xticklabels(properties)
    ax.set_title("GLCM Texture Properties Comparison")
    ax.legend()

    st.pyplot(fig)

    # Show full table below
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Query Texture Values**")
        st.table({k: [round(v, 4)] for k, v in query_stats.items()})
    with col4:
        st.markdown("**Retrieved Texture Values**")
        st.table({k: [round(v, 4)] for k, v in result_stats.items()})


def show_hu_analysis(query_image_path, result_image_path):
    st.subheader("Shape Analysis (Hu Moments)")

    query_binary = get_hu_binary_image(query_image_path)
    result_binary = get_hu_binary_image(result_image_path)
    query_hu = extract_hu_moments(query_image_path)
    result_hu = extract_hu_moments(result_image_path)

    if query_binary is None or result_binary is None or query_hu is None or result_hu is None:
        st.warning("Unable to compute Hu Moments analysis.")
        return

    fig1, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(query_binary, cmap="gray")
    axes[0].set_title("Query Binary Shape")
    axes[0].axis("off")

    axes[1].imshow(result_binary, cmap="gray")
    axes[1].set_title("Retrieved Binary Shape")
    axes[1].axis("off")
    st.pyplot(fig1)

    fig2, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(1, 8)
    ax.bar(x - 0.2, query_hu, width=0.4, label="Query")
    ax.bar(x + 0.2, result_hu, width=0.4, label="Retrieved")
    ax.set_xticks(x)
    ax.set_title("Hu Moments Comparison")
    ax.set_xlabel("Hu Moment Index")
    ax.legend()
    st.pyplot(fig2)


def show_orb_analysis(query_image_path, result_image_path):
    st.subheader("Keypoint Analysis (ORB)")

    query_gray, kp1, des1 = extract_orb_keypoints_and_descriptors(query_image_path)
    result_gray, kp2, des2 = extract_orb_keypoints_and_descriptors(result_image_path)

    if query_gray is None or result_gray is None or des1 is None or des2 is None:
        st.warning("Unable to compute ORB keypoint analysis.")
        return

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = [m for m in matches if m.distance < 50]
    top_matches = good_matches[:30]

    query_rgb = cv2.cvtColor(query_gray, cv2.COLOR_GRAY2RGB)
    result_rgb = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2RGB)

    query_kp_img = cv2.drawKeypoints(query_rgb, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    result_kp_img = cv2.drawKeypoints(result_rgb, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    fig1, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(query_kp_img)
    axes[0].set_title(f"Query Keypoints: {len(kp1)}")
    axes[0].axis("off")

    axes[1].imshow(result_kp_img)
    axes[1].set_title(f"Retrieved Keypoints: {len(kp2)}")
    axes[1].axis("off")
    st.pyplot(fig1)

    match_img = cv2.drawMatches(
        query_rgb, kp1, result_rgb, kp2, top_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    st.image(match_img, caption=f"Top ORB Matches: {len(top_matches)}", use_container_width=True)


def show_algorithm_explanation(algorithm):
    st.subheader("Feature Explanation")

    if algorithm == "Color Histogram":
        st.info(
            "Color Histogram retrieves images based on similarity in color distribution. "
            "It works best when the query image and retrieved image share similar overall colors."
        )
    elif algorithm == "GLCM Texture":
        st.info(
            "GLCM retrieves images based on texture characteristics such as contrast, correlation, energy, and homogeneity. "
            "It is useful for comparing surface patterns and repeated textures."
        )
    elif algorithm == "Hu Moments":
        st.info(
            "Hu Moments retrieve images based on overall shape structure. "
            "It is useful when object silhouette and global form are important."
        )
    elif algorithm == "ORB":
        st.info(
            "ORB retrieves images based on local keypoints and descriptor matches. "
            "It is effective for structured objects with clear edges, corners, and details."
        )


# =========================================================
# Sidebar Controls
# =========================================================
st.sidebar.header("Search Settings")

algorithm = st.sidebar.selectbox(
    "Choose Algorithm",
    ["Color Histogram", "GLCM Texture", "Hu Moments", "ORB"]
)

top_k = st.sidebar.selectbox(
    "Top-K Results",
    [5, 10, 15],
    index=0
)

threshold = st.sidebar.slider(
    "Similarity Threshold (%)",
    min_value=0,
    max_value=100,
    value=50,
    step=10
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Query Image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# Main App
# =========================================================
if uploaded_file is not None:
    uploaded_bytes = uploaded_file.read()

    # keep uploaded bytes in session
    st.session_state.uploaded_image_bytes = uploaded_bytes

    # always rebuild a temp query file from the saved uploaded bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(st.session_state.uploaded_image_bytes)
        query_image_path = tmp_file.name

    st.subheader("Query Image")
    query_img = Image.open(query_image_path)
    st.image(query_img, caption="Uploaded Query Image", width=300)

    if st.button("Retrieve Similar Images"):
        with st.spinner("Processing image and retrieving results..."):
            if algorithm == "Color Histogram":
                results = retrieve_by_similarity(query_image_path, feature_type="color", top_k=top_k)
            elif algorithm == "GLCM Texture":
                results = retrieve_by_similarity(query_image_path, feature_type="glcm", top_k=top_k)
            elif algorithm == "Hu Moments":
                results = retrieve_by_similarity(query_image_path, feature_type="hu", top_k=top_k)
            else:
                results = retrieve_by_orb(query_image_path, top_k=top_k)

            filtered_results = filter_results_by_threshold(results, threshold, algorithm)

            st.session_state.retrieved_results = results
            st.session_state.filtered_results = filtered_results
            st.session_state.selected_algorithm = algorithm
            st.session_state.query_image_path = query_image_path

    # IMPORTANT: these lines must be OUTSIDE the button block
    filtered_results = st.session_state.filtered_results
    saved_query_image_path = st.session_state.query_image_path
    saved_algorithm = st.session_state.selected_algorithm

    if filtered_results is not None:
        st.subheader("Retrieved Results")

        if len(filtered_results) == 0:
            st.warning("No results found above the selected similarity threshold.")
        else:
            cols_per_row = 5

            for i in range(0, len(filtered_results), cols_per_row):
                row_results = filtered_results[i:i + cols_per_row]
                cols = st.columns(cols_per_row)

                for j, result in enumerate(row_results, start=i):
                    with cols[j - i]:
                        result_path = normalize_cloud_path(result["image_path"])
                        result_img = Image.open(result_path)
                        st.image(result_img, use_container_width=True)

                        st.write(f"**Category:** {result['label']}")

                        if saved_algorithm == "ORB":
                            display_score = result.get("normalized_score", 0)
                            st.write(f"**Similarity:** {display_score:.2f}%")
                            st.write(f"**Raw Score:** {int(result['score'])}")
                        else:
                            st.write(f"**Similarity:** {result['score'] * 100:.2f}%")

                        image_bytes = image_to_bytes(normalize_cloud_path(result["image_path"]))
                        st.download_button(
                            label="Download Image",
                            data=image_bytes,
                            file_name=os.path.basename(result["image_path"]),
                            mime="image/jpeg",
                            key=f"download_{j}"
                        )

            zip_data = create_zip_from_results(filtered_results)
            st.download_button(
                label="Download All Retrieved Images (ZIP)",
                data=zip_data,
                file_name="retrieved_images.zip",
                mime="application/zip",
                key="download_zip"
            )

            st.subheader("Retrieval Summary")
            labels = [r["label"] for r in filtered_results]
            unique_labels, counts = np.unique(labels, return_counts=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Retrieved", len(filtered_results))
            with col2:
                st.metric("Unique Categories", len(unique_labels))
            with col3:
                if saved_algorithm == "ORB":
                    avg_score = np.mean([r.get("normalized_score", 0) for r in filtered_results])
                else:
                    avg_score = np.mean([r["score"] * 100 for r in filtered_results])
                st.metric("Average Similarity", f"{avg_score:.2f}%")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(unique_labels, counts)
            ax.set_title("Retrieved Image Distribution by Category")
            ax.set_ylabel("Count")
            ax.set_xlabel("Category")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            show_algorithm_explanation(saved_algorithm)

            top_result = filtered_results[0]["image_path"]

            if saved_algorithm == "Color Histogram":
                show_color_analysis(saved_query_image_path, top_result)
            elif saved_algorithm == "GLCM Texture":
                show_glcm_analysis(saved_query_image_path, top_result)
            elif saved_algorithm == "Hu Moments":
                show_hu_analysis(saved_query_image_path, top_result)
            elif saved_algorithm == "ORB":
                show_orb_analysis(saved_query_image_path, top_result)

else:
    st.info("Please upload an image from the sidebar to begin retrieval.")


# =========================================================
# Footer
# =========================================================
st.markdown("---")
st.caption("CBIR System using Color Histogram, GLCM, Hu Moments, and ORB")
