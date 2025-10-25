import streamlit as st
import os
from PIL import Image
from pathlib import Path
import shutil
import numpy as np

from Feature_Extraction.Feature_Extractions import find_duplicates


st.set_page_config(
    page_title="DoppelHash",
    page_icon="../assets/icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

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


def save_uploaded_files(uploaded_files):
    """Save uploaded files to a temporary directory"""
    #todo: give choice to user store file filtered from duplicates
    
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



# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
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
    
    if st.session_state.processed and st.button(" Reset"):
        st.session_state.duplicates = []
        st.session_state.processed = False
        st.session_state.uploader_key += 1
        if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
            shutil.rmtree(st.session_state.temp_dir)
        st.rerun()


col1, col2 = st.columns([1, 10])


with col1:
    st.image("../assets/icon.png", width=80)
with col2:
    st.markdown("<h1 style='margin-top: 0;, margin-lef: 0;'>DoppelHash</h1>", unsafe_allow_html=True)

st.markdown("### Find and manage duplicate images with ease!")




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
    st.subheader("🚀 Actions")
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files uploaded")
        
        if st.button("Find Duplicates", type="primary"):
            try:
                temp_dir = save_uploaded_files(uploaded_files)
                st.session_state.temp_dir = temp_dir
                
                with st.spinner("Analyzing images..."):
                    duplicates, num_groups = find_duplicates(temp_dir, algorithm, threshold)
                    st.session_state.duplicates = duplicates
                    st.session_state.total_files = len(uploaded_files)
                    st.session_state.processed = True
                
                st.success("✨ Processing complete!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.info("Upload images to get started")

# Display results
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
        st.subheader("Duplicate Groups")
        
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
                
                # images in grid
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
        st.success("🎉 No duplicates found! All your images are unique.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 3rem;'>
        <p>Achouak Guennoun & Hajar Makhlouf | DoppelHash v1.0</p>
    </div>
    """, unsafe_allow_html=True)