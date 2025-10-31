import streamlit as st
import os
from PIL import Image
from pathlib import Path
import shutil
import numpy as np
import sys
import zipfile
import io
import tempfile
import time

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.Feature_Extractions import find_duplicates


st.set_page_config(
    page_title="DoppelHash",
    page_icon="src/UI/assets/icon.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    .stAppToolbar {display: none;}
    [data-testid="stToolbar"] {display: none;}
    [data-testid="manage-app-button"] {display: none;}
    .viewerBadge_container__1QSob {display: none;}
    </style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .duplicate-group {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #ff4b4b;
    }
    .similarity-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .high-similarity {
        background-color: #ff4b4b;
        color: white;
    }
    .medium-similarity {
        background-color: #ffa726;
        color: white;
    }
    .low-similarity {
        background-color: #66bb6a;
        color: white;
    }
    .tab-badge {
        background-color: #2196F3;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 10px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

if 'duplicates' not in st.session_state:
    st.session_state.duplicates = []
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0
if 'stats' not in st.session_state:
    st.session_state.stats = {}


def save_uploaded_files(uploaded_files):
    """Save uploaded files to a temporary directory"""
    temp_dir = "temp_uploaded_images"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Uploading: {uploaded_file.name}")

        safe_filename = os.path.basename(uploaded_file.name.replace('\\', '/'))
        file_path = os.path.join(temp_dir, safe_filename)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        progress_bar.progress((idx + 1) / len(uploaded_files))
    
    progress_bar.empty()
    status_text.empty()
    
    return temp_dir

def get_similarity_badge_class(similarity):
    """Get CSS class based on similarity score"""
    if similarity >= 90:
        return "high-similarity"
    elif similarity >= 75:
        return "medium-similarity"
    else:
        return "low-similarity"

def calculate_space_wasted(duplicate_groups, temp_dir):
    """Calculate total wasted space from duplicates"""
    total_wasted = 0
    
    for group_data in duplicate_groups:
        group = group_data['group']
        sizes = []
        
        for img_name in group:
            img_path = os.path.join(temp_dir, img_name)
            if os.path.exists(img_path):
                sizes.append(os.path.getsize(img_path))
        
        if sizes:
            total_wasted += sum(sizes) - max(sizes)
    
    return total_wasted


def process_zip_folder(uploaded_zip, algorithm, threshold, sim_method, num_bands, rows_per_band):
    """Process ZIP folder and return filtered ZIP"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            extract_path = temp_path / "extracted"
            filtered_path = temp_path / "filtered"
            extract_path.mkdir()
            filtered_path.mkdir()
            
            # Extract ZIP
            with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as zip_ref:
                zip_ref.extractall(extract_path)
            
            image_extensions = {'.png', '.jpg', '.jpeg'}
            image_files = []
            seen_files = set()
            
            for file_path in extract_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    abs_path = file_path.resolve()
                    if abs_path not in seen_files:
                        image_files.append(file_path)
                        seen_files.add(abs_path)
            
            if not image_files:
                return None, "No images found in the ZIP file!", 0, 0, 0
            
            
            
            # Create a flat temp folder for find_duplicates
            flat_temp = temp_path / "flat_temp"
            flat_temp.mkdir()
            
            file_mapping = {}
            for idx, img_file in enumerate(image_files):
                # Create unique flat name
                flat_name = f"{idx}_{img_file.name}"
                flat_path = flat_temp / flat_name
                shutil.copy2(img_file, flat_path)
                file_mapping[flat_name] = img_file
            
            # Find duplicates in flat folder
            result = find_duplicates(
                str(flat_temp), 
                algorithm, 
                threshold,
                sim_method=sim_method,
                num_bands=num_bands,
                rows_per_band=rows_per_band
            )
            
            if not result or result == []:
                duplicates = []
                num_groups = 0
            elif isinstance(result, tuple) and len(result) >= 2:
                duplicates = result[0]
                num_groups = result[1]
            else:
                duplicates = result if isinstance(result, list) else []
                num_groups = len(duplicates)
            
            files_to_remove = set()
            if duplicates:
                for group_data in duplicates:
                    dup_group = group_data['group']

                    original_paths = [file_mapping[flat_name] for flat_name in dup_group]
                    
                    files_to_remove.update(original_paths[1:])
            
            kept_files = 0
            removed_files = len(files_to_remove)
            
            for img_file in image_files:
                if img_file not in files_to_remove:
                    relative_path = img_file.relative_to(extract_path)
                    dest_file = filtered_path / relative_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_file, dest_file)
                    kept_files += 1
            
            # Create ZIP of filtered folder
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file_path in filtered_path.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(filtered_path)
                        zip_file.write(file_path, arcname)
            
            zip_buffer.seek(0)
            return zip_buffer, None, len(image_files), removed_files, kept_files
            
    except Exception as e:
        return None, f"Error processing folder: {str(e)}", 0, 0, 0


with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader(" Detection Method")
    
    sim_method = st.radio(
        "Choose method:",
        options=['Bruteforce', 'lsh'],
        format_func=lambda x: {
            'Bruteforce': 'Bruteforce',
            'lsh': 'LSH'
        }[x],
    )
    num_bands=8
    rows_per_band=8
    if sim_method == 'lsh':
        with st.expander("⚙️ LSH Settings", expanded=False):
            valid_bands = [1, 2, 4, 8, 16, 32, 64]
            
            num_bands = st.select_slider(
                "Number of Bands",
                options=valid_bands,
                value=8
            )
            
            rows_per_band = 64 // num_bands
            
            st.caption(f"Hash size: {num_bands * rows_per_band} bits ({num_bands} bands × {rows_per_band} rows/band)")

    
    algorithm = st.selectbox(
        "Hash Algorithm",
        ["phash"],
    )
    
    threshold = st.slider(
        "Similarity Threshold (%)",
        min_value=50,
        max_value=100,
        value=85,
        step=5,
    )
    
    st.markdown("---")
    
    if st.session_state.processed and st.button("Reset"):
        st.session_state.duplicates = []
        st.session_state.processed = False
        st.session_state.stats = {}
        st.session_state.uploader_key += 1
        if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
            shutil.rmtree(st.session_state.temp_dir)
        st.rerun()


col1, col2 = st.columns([1, 10])

with col1:
    st.image("src/UI/assets/icon.svg", width=80)
with col2:
    st.markdown("<h1 style='margin-top: 0;, margin-left: 0;'>DoppelHash</h1>", unsafe_allow_html=True)

st.markdown("### Find and manage duplicate images with ease!")

tab1, tab2 = st.tabs(["🔍 Find Duplicates", "📁 Filter Folder"])

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📁 Upload Images")
        uploaded_files = st.file_uploader(
            "Choose folder",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files="directory",
            key=f"file_uploader_{st.session_state.uploader_key}"
        )

    with col2:
        st.subheader("Actions")
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files uploaded")
            
            if st.button("Find Duplicates", type="primary", key="find_btn"):
                try:
                    temp_dir = save_uploaded_files(uploaded_files)
                    st.session_state.temp_dir = temp_dir
                    
                    with st.spinner("Analyzing images..."):
                        result = find_duplicates(
                            temp_dir,
                            algorithm,
                            threshold,
                            sim_method=sim_method,
                            num_bands=num_bands,
                            rows_per_band=rows_per_band
                        )
                        
                        if not result or result == []:
                            duplicates = []
                            num_groups = 0
                            stats = {}
                        elif isinstance(result, tuple) and len(result) == 3:
                            duplicates, num_groups, stats = result
                        elif isinstance(result, tuple) and len(result) == 2:
                            duplicates, num_groups = result
                            stats = {}
                        else:
                            duplicates = result if isinstance(result, list) else []
                            num_groups = len(duplicates)
                            stats = {}
                        
                        st.session_state.duplicates = duplicates
                        st.session_state.total_files = len(uploaded_files)
                        st.session_state.stats = stats
                        st.session_state.processed = True
                    
                    st.success("✨ Processing complete!")
                    
                    # Display performance stats if available
                    with col1:
                        if stats:
                            st.markdown("---")
                            st.subheader("⚡ Performance")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Method", stats['method'])
                            with col2:
                                if stats['method'] == 'lsh':
                                    time_val = stats['comparison_time_lsh']
                                else:
                                    time_val = stats['comparison_time_brute']
                                if time_val < 0.01:
                                    st.metric("Total Time", f"{time_val*1000:.2f}ms")
                                else:
                                    st.metric("Total Time", f"{time_val:.4f}s")
                                    
                            with col3:
                                st.metric("Comparisons", f"{stats['comparisons_made']:,}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.info("Upload images to get started")

    if st.session_state.processed:
        st.markdown("---")
        st.header("📊 Results")
        
        num_duplicate_groups = len(st.session_state.duplicates)
        total_duplicates = sum(len(group['group']) for group in st.session_state.duplicates)
        
        if num_duplicate_groups > 0:
            avg_similarity = np.mean([group['avg_similarity'] for group in st.session_state.duplicates])
            space_wasted = calculate_space_wasted(st.session_state.duplicates, st.session_state.temp_dir)
        else:
            avg_similarity = 0
            space_wasted = 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Images", st.session_state.total_files)
        with col2:
            st.metric("Duplicate Groups", num_duplicate_groups)
        with col3:
            st.metric("Total Duplicates", total_duplicates)
        with col4:
            st.metric("Space Wasted", f"{space_wasted / 1024 / 1024:.2f} MB")

        
        if num_duplicate_groups > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Similarity", f"{avg_similarity:.1f}%")
            with col2:
                unique_images = st.session_state.total_files - (total_duplicates - num_duplicate_groups)
                st.metric("Unique Images", unique_images)
            with col3:
                reduction = (total_duplicates - num_duplicate_groups) / st.session_state.total_files * 100
                st.metric("Potential Reduction", f"{reduction:.1f}%")
        
        if num_duplicate_groups > 0:
            st.markdown("---")
            st.subheader("🔎 Duplicate Groups")
            
            # sort groups by similarity desc
            sorted_groups = sorted(st.session_state.duplicates, key=lambda x: x['avg_similarity'], reverse=True)
            
            for idx, group_data in enumerate(sorted_groups, 1):
                group = group_data['group']
                avg_sim = group_data['avg_similarity']
                badge_class = get_similarity_badge_class(avg_sim)
                
                header_html = f"""
                <div style='display: flex; align-items: center; gap: 10px;'>
                    <span style='font-size: 1.1rem; font-weight: bold;'>📂 Group {idx}</span>
                    <span class='similarity-badge {badge_class}'>{avg_sim}% Similar</span>
                    <span style='color: #666;'>({len(group)} images)</span>
                </div>
                """
                
                with st.expander(f"Group {idx} - {avg_sim}% similarity - {len(group)} images", expanded=(idx == 1)):
                    st.markdown(header_html, unsafe_allow_html=True)
                    st.markdown("")
                    
                    cols = st.columns(min(len(group), 4))
                    
                    for i, img_name in enumerate(group):
                        col_idx = i % 4
                        with cols[col_idx]:
                            try:
                                img_path = os.path.join(st.session_state.temp_dir, img_name)
                                img = Image.open(img_path)
                                st.image(img, caption=img_name, use_container_width=True)
                                
                                file_size = os.path.getsize(img_path) / 1024
                                st.caption(f"📏 {file_size:.1f} KB")
                                st.caption(f"📐 {img.size[0]}x{img.size[1]}")
                            except Exception as e:
                                st.error(f"Error loading {img_name}: {e}")
        else:
            st.success("No duplicates found! All images are unique.")

with tab2:
    st.subheader("📁 Filter Folder from Duplicates")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_zip = st.file_uploader(
            "Upload ZIP file",
            type=['zip'],
            key="zip_uploader",
        )
    
    
    if uploaded_zip:
        with col2:
            st.success(f"✅ ZIP file uploaded: {uploaded_zip.name}")
        
        if st.button("Filter Folder", type="primary", key="filter_btn"):
            with st.spinner("🔄 Processing folder... This may take a moment."):
                zip_buffer, error, total_images, removed, kept = process_zip_folder(
                    uploaded_zip, algorithm, threshold, sim_method, num_bands, rows_per_band
                )
                
                if error:
                    st.error(f"❌ {error}")
                else:
                    with col2:
                        st.success("✨ Processing complete!")
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📸 Original Images", total_images)
                    with col2:
                        st.metric("🗑️ Duplicates Removed", removed)
                    with col3:
                        st.metric("✅ Images Kept", kept)
                    
                    if removed > 0:
                        reduction_pct = (removed / total_images) * 100
                        st.info(f"💾 Space saved: {reduction_pct:.1f}% reduction in image count")
                    
                    st.download_button(
                        label="📥 Download Filtered Folder",
                        data=zip_buffer,
                        file_name=f"filtered_{uploaded_zip.name}",
                        mime="application/zip",
                        type="primary",
                        use_container_width=True
                    )
    else:
        st.info("📤 Upload a ZIP file to get started")


st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 3rem;'>
        <p> DoppelHash v1.0</p>
    </div>
    """, unsafe_allow_html=True)