import streamlit as st
import numpy as np
import cv2
from PIL import Image
import open3d as o3d
import os
import tempfile
import time
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
import plotly.express as px

# Configure Streamlit page
st.set_page_config(
    page_title="AI 3D Model Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Meshy AI-like styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    .hero-header {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f5ff 0%, #ff00ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        font-family: 'Arial Black', sans-serif;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: rgba(255,255,255,0.8);
        margin-bottom: 0;
    }
    
    .feature-card {
        background: rgba(255,255,255,0.1);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    .model-preview {
        background: rgba(0,0,0,0.3);
        border-radius: 15px;
        padding: 1rem;
        border: 2px solid rgba(255,255,255,0.1);
    }
    
    .success-glow {
        background: linear-gradient(135deg, rgba(0,255,127,0.2) 0%, rgba(0,255,255,0.2) 100%);
        border: 2px solid rgba(0,255,127,0.5);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .processing-card {
        background: linear-gradient(135deg, rgba(255,165,0,0.2) 0%, rgba(255,69,0,0.2) 100%);
        border: 2px solid rgba(255,165,0,0.5);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .viewer-controls {
        background: rgba(255,255,255,0.05);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stSelectbox > div > div {
        background-color: rgba(255,255,255,0.1);
        border-radius: 10px;
    }
    
    .stSlider > div > div {
        background-color: rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

def preprocess_image_for_3d(image, enhancement_type="edge_enhanced"):
    """
    Advanced image preprocessing for better 3D reconstruction
    """
    image_array = np.array(image)
    
    if len(image_array.shape) == 3:
        if image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        color = image_array
    else:
        gray = image_array
        color = np.stack([gray, gray, gray], axis=-1)
    
    # Apply different enhancement techniques
    if enhancement_type == "edge_enhanced":
        # Enhance edges for better 3D structure
        edges = cv2.Canny(gray, 50, 150)
        gray = cv2.addWeighted(gray, 0.8, edges, 0.2, 0)
        
    elif enhancement_type == "smooth_terrain":
        # Apply Gaussian smoothing for terrain-like surfaces
        gray = gaussian_filter(gray, sigma=1.5)
        
    elif enhancement_type == "sharp_details":
        # Enhance contrast and details
        gray = cv2.equalizeHist(gray)
        
    elif enhancement_type == "artistic":
        # Create more dramatic height variations
        gray = np.power(gray / 255.0, 0.7) * 255
        gray = gray.astype(np.uint8)
    
    return gray, color

def generate_mesh_from_pointcloud(points, colors, mesh_quality="medium"):
    """
    Generate a mesh from point cloud using different algorithms
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    if mesh_quality == "point_cloud":
        return pcd, "point_cloud"
    
    try:
        # Estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
        
        if mesh_quality == "low":
            # Simple Poisson reconstruction
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6)
        elif mesh_quality == "medium": 
            # Better Poisson reconstruction
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        else:  # high quality
            # High-quality Poisson reconstruction
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
        
        # Clean up the mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        return mesh, "mesh"
        
    except Exception as e:
        st.warning(f"Mesh generation failed: {e}. Returning point cloud instead.")
        return pcd, "point_cloud"

def create_advanced_3d_model(image, height_scale, enhancement_type, mesh_quality, density_factor):
    """
    Create advanced 3D model with various options like Meshy AI
    """
    
    # Preprocess image
    gray, color = preprocess_image_for_3d(image, enhancement_type)
    
    # Calculate downsample factor based on density
    original_h, original_w = gray.shape
    if density_factor == "ultra_high":
        downsample = 1
    elif density_factor == "high":
        downsample = max(1, max(original_w, original_h) // 800)
    elif density_factor == "medium":
        downsample = max(1, max(original_w, original_h) // 500)
    elif density_factor == "low":
        downsample = max(1, max(original_w, original_h) // 300)
    else:  # preview
        downsample = max(1, max(original_w, original_h) // 150)
    
    # Downsample
    if downsample > 1:
        gray = gray[::downsample, ::downsample]
        color = color[::downsample, ::downsample]
    
    height, width = gray.shape
    
    # Create coordinate grids
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    
    # Advanced height mapping
    z_coords = gray.astype(np.float32) / 255.0 * height_scale
    
    # Add some randomness for more organic look (optional)
    if enhancement_type == "artistic":
        noise = np.random.normal(0, height_scale * 0.02, z_coords.shape)
        z_coords += noise
    
    # Flatten arrays
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten() 
    z_flat = z_coords.flatten()
    
    points = np.column_stack((x_flat, y_flat, z_flat))
    colors_flat = color.reshape(-1, 3) / 255.0
    
    # Generate 3D model (mesh or point cloud)
    model, model_type = generate_mesh_from_pointcloud(points, colors_flat, mesh_quality)
    
    stats = {
        'original_dimensions': f"{original_w} × {original_h}",
        'processed_dimensions': f"{width} × {height}", 
        'total_points': len(points),
        'model_type': model_type,
        'enhancement': enhancement_type,
        'mesh_quality': mesh_quality,
        'density': density_factor,
        'downsample_factor': downsample,
        'height_range': f"{z_flat.min():.2f} to {z_flat.max():.2f}"
    }
    
    return model, points, colors_flat, stats

def create_enhanced_3d_visualization(points, colors, view_mode="point_cloud", color_mode="original"):
    """Create enhanced interactive 3D visualization with multiple view modes"""
    
    # Sample points for performance if too many
    max_points = 8000 if view_mode == "point_cloud" else 5000
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points_sample = points[indices]
        colors_sample = colors[indices]
    else:
        points_sample = points
        colors_sample = colors
    
    # Color processing based on mode
    if color_mode == "original":
        colors_rgb = colors_sample
    elif color_mode == "height":
        # Color by height (elevation)
        z_norm = (points_sample[:, 2] - points_sample[:, 2].min()) / (points_sample[:, 2].max() - points_sample[:, 2].min())
        colors_rgb = px.colors.sample_colorscale('viridis', z_norm)
        colors_rgb = np.array([[int(c[4:6], 16)/255, int(c[6:8], 16)/255, int(c[8:10], 16)/255] for c in colors_rgb])
    elif color_mode == "terrain":
        # Terrain-like coloring
        z_norm = (points_sample[:, 2] - points_sample[:, 2].min()) / (points_sample[:, 2].max() - points_sample[:, 2].min())
        colors_rgb = px.colors.sample_colorscale('terrain', z_norm)
        colors_rgb = np.array([[int(c[4:6], 16)/255, int(c[6:8], 16)/255, int(c[8:10], 16)/255] for c in colors_rgb])
    else:  # grayscale
        gray_vals = np.mean(colors_sample, axis=1)
        colors_rgb = np.column_stack([gray_vals, gray_vals, gray_vals])
    
    # Convert colors to format for Plotly
    colors_plotly = ['rgb({},{},{})'.format(
        int(c[0]*255), int(c[1]*255), int(c[2]*255)
    ) for c in colors_rgb]
    
    fig = go.Figure()
    
    if view_mode == "point_cloud":
        # Point cloud view
        fig.add_trace(go.Scatter3d(
            x=points_sample[:, 0],
            y=points_sample[:, 1], 
            z=points_sample[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=colors_plotly,
                opacity=0.8,
                line=dict(width=0)
            ),
            name='Point Cloud',
            hovertemplate='<b>Position</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Height: %{z:.1f}<extra></extra>'
        ))
    
    elif view_mode == "surface":
        # Surface view using triangulation
        try:
            # Create a regular grid for surface plot
            x_unique = np.unique(points_sample[:, 0])
            y_unique = np.unique(points_sample[:, 1])
            
            if len(x_unique) > 3 and len(y_unique) > 3:
                # Interpolate to regular grid
                from scipy.interpolate import griddata
                
                # Sample fewer points for surface
                if len(points_sample) > 2000:
                    indices = np.random.choice(len(points_sample), 2000, replace=False)
                    points_surf = points_sample[indices]
                    colors_surf = colors_rgb[indices]
                else:
                    points_surf = points_sample
                    colors_surf = colors_rgb
                
                # Create regular grid
                xi = np.linspace(points_surf[:, 0].min(), points_surf[:, 0].max(), 50)
                yi = np.linspace(points_surf[:, 1].min(), points_surf[:, 1].max(), 50)
                xi_grid, yi_grid = np.meshgrid(xi, yi)
                
                # Interpolate z values
                zi_grid = griddata((points_surf[:, 0], points_surf[:, 1]), points_surf[:, 2], 
                                 (xi_grid, yi_grid), method='linear', fill_value=0)
                
                # Interpolate colors
                colors_r = griddata((points_surf[:, 0], points_surf[:, 1]), colors_surf[:, 0], 
                                   (xi_grid, yi_grid), method='linear', fill_value=0)
                colors_g = griddata((points_surf[:, 0], points_surf[:, 1]), colors_surf[:, 1], 
                                   (xi_grid, yi_grid), method='linear', fill_value=0)
                colors_b = griddata((points_surf[:, 0], points_surf[:, 1]), colors_surf[:, 2], 
                                   (xi_grid, yi_grid), method='linear', fill_value=0)
                
                # Create surface
                fig.add_trace(go.Surface(
                    x=xi_grid, y=yi_grid, z=zi_grid,
                    surfacecolor=np.stack([colors_r, colors_g, colors_b], axis=-1),
                    colorscale=[[0, 'rgb(0,0,0)'], [1, 'rgb(255,255,255)']],
                    showscale=False,
                    name='Surface',
                    hovertemplate='<b>Surface</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Height: %{z:.1f}<extra></extra>'
                ))
            else:
                # Fallback to point cloud if surface can't be created
                fig.add_trace(go.Scatter3d(
                    x=points_sample[:, 0], y=points_sample[:, 1], z=points_sample[:, 2],
                    mode='markers', marker=dict(size=3, color=colors_plotly, opacity=0.8),
                    name='Points (Surface Failed)'
                ))
        except Exception as e:
            # Fallback to point cloud
            fig.add_trace(go.Scatter3d(
                x=points_sample[:, 0], y=points_sample[:, 1], z=points_sample[:, 2],
                mode='markers', marker=dict(size=3, color=colors_plotly, opacity=0.8),
                name='Points (Surface Failed)'
            ))
    
    elif view_mode == "wireframe":
        # Wireframe view - create a mesh outline
        try:
            # Sample fewer points for wireframe
            if len(points_sample) > 1000:
                indices = np.random.choice(len(points_sample), 1000, replace=False)
                points_wire = points_sample[indices]
            else:
                points_wire = points_sample
            
            # Create triangulation
            from scipy.spatial import Delaunay
            points_2d = points_wire[:, :2]
            tri = Delaunay(points_2d)
            
            # Create wireframe lines
            lines_x, lines_y, lines_z = [], [], []
            for triangle in tri.simplices:
                for i in range(3):
                    j = (i + 1) % 3
                    lines_x.extend([points_wire[triangle[i], 0], points_wire[triangle[j], 0], None])
                    lines_y.extend([points_wire[triangle[i], 1], points_wire[triangle[j], 1], None])
                    lines_z.extend([points_wire[triangle[i], 2], points_wire[triangle[j], 2], None])
            
            fig.add_trace(go.Scatter3d(
                x=lines_x, y=lines_y, z=lines_z,
                mode='lines',
                line=dict(color='rgba(255,255,255,0.6)', width=2),
                name='Wireframe',
                hoverinfo='skip'
            ))
            
            # Add some points for reference
            fig.add_trace(go.Scatter3d(
                x=points_wire[:, 0], y=points_wire[:, 1], z=points_wire[:, 2],
                mode='markers',
                marker=dict(size=2, color=colors_plotly[:len(points_wire)] if len(colors_plotly) >= len(points_wire) else 'white', opacity=0.8),
                name='Vertices',
                hovertemplate='<b>Vertex</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Height: %{z:.1f}<extra></extra>'
            ))
        except Exception as e:
            # Fallback to point cloud
            fig.add_trace(go.Scatter3d(
                x=points_sample[:, 0], y=points_sample[:, 1], z=points_sample[:, 2],
                mode='markers', marker=dict(size=3, color=colors_plotly, opacity=0.8),
                name='Points (Wireframe Failed)'
            ))
    
    # Enhanced layout
    fig.update_layout(
        title=dict(
            text=f"Interactive 3D Model - {view_mode.title()} View",
            font=dict(size=20, color="white"),
            x=0.5
        ),
        scene=dict(
            xaxis=dict(
                title="X Axis",
                backgroundcolor="rgba(0,0,0,0.1)",
                gridcolor="rgba(255,255,255,0.2)",
                showbackground=True,
                zerolinecolor="rgba(255,255,255,0.4)"
            ),
            yaxis=dict(
                title="Y Axis",
                backgroundcolor="rgba(0,0,0,0.1)",
                gridcolor="rgba(255,255,255,0.2)",
                showbackground=True,
                zerolinecolor="rgba(255,255,255,0.4)"
            ),
            zaxis=dict(
                title="Height",
                backgroundcolor="rgba(0,0,0,0.1)",
                gridcolor="rgba(255,255,255,0.2)",
                showbackground=True,
                zerolinecolor="rgba(255,255,255,0.4)"
            ),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='cube'
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig

def create_side_by_side_view(points, colors):
    """Create side-by-side comparison views"""
    from plotly.subplots import make_subplots
    
    # Sample points
    if len(points) > 3000:
        indices = np.random.choice(len(points), 3000, replace=False)
        points_sample = points[indices]
        colors_sample = colors[indices]
    else:
        points_sample = points
        colors_sample = colors
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('Original Colors', 'Height Map'),
        horizontal_spacing=0.05
    )
    
    # Original colors view
    colors_orig = ['rgb({},{},{})'.format(
        int(c[0]*255), int(c[1]*255), int(c[2]*255)
    ) for c in colors_sample]
    
    fig.add_trace(
        go.Scatter3d(
            x=points_sample[:, 0], y=points_sample[:, 1], z=points_sample[:, 2],
            mode='markers',
            marker=dict(size=2, color=colors_orig, opacity=0.8),
            name='Original',
            hovertemplate='Original<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Height: %{z:.1f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Height-colored view
    z_norm = (points_sample[:, 2] - points_sample[:, 2].min()) / (points_sample[:, 2].max() - points_sample[:, 2].min())
    
    fig.add_trace(
        go.Scatter3d(
            x=points_sample[:, 0], y=points_sample[:, 1], z=points_sample[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=points_sample[:, 2],
                colorscale='viridis',
                opacity=0.8,
                showscale=True,
                colorbar=dict(title="Height", x=1.02)
            ),
            name='Height Map',
            hovertemplate='Height Map<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Height: %{z:.1f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Side-by-Side Comparison",
        font=dict(color="white"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=500
    )
    
    # Update 3D scenes
    scene_props = dict(
        bgcolor="rgba(0,0,0,0)",
        xaxis=dict(backgroundcolor="rgba(0,0,0,0.1)", gridcolor="rgba(255,255,255,0.2)"),
        yaxis=dict(backgroundcolor="rgba(0,0,0,0.1)", gridcolor="rgba(255,255,255,0.2)"),
        zaxis=dict(backgroundcolor="rgba(0,0,0,0.1)", gridcolor="rgba(255,255,255,0.2)"),
        camera=dict(eye=dict(x=1.2, y=1.2, z=1)),
        aspectmode='cube'
    )
    
    fig.update_scenes(scene_props)
    
    return fig

def save_model_to_bytes(model, model_type, filename_base):
    """Save 3D model to bytes for download"""
    downloads = {}
    
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        try:
            o3d.io.write_point_cloud(tmp_file.name, model)
            with open(tmp_file.name, 'rb') as f:
                downloads['ply'] = f.read()
        finally:
            os.unlink(tmp_file.name)
    
    # If it's a mesh, also save as OBJ
    if model_type == "mesh":
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp_file:
            try:
                o3d.io.write_triangle_mesh(tmp_file.name, model)
                with open(tmp_file.name, 'rb') as f:
                    downloads['obj'] = f.read()
            except:
                pass  # OBJ export might fail, PLY is always available
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    return downloads

def main():
    # Hero Section
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🤖 AI 3D Generator</div>
        <div class="hero-subtitle">Transform any image into stunning 3D models with advanced AI processing</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main interface
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        st.markdown("### 🎨 Upload & Configure")
        
        # File uploader with drag and drop
        uploaded_file = st.file_uploader(
            "Drop your image here or click to browse",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
            help="Supported formats: PNG, JPG, JPEG, BMP, TIFF, WebP"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            
            # Image preview
            st.image(image, caption=f"📸 {uploaded_file.name}", use_column_width=True)
            
            # Image analysis
            w, h = image.size
            aspect_ratio = w / h
            megapixels = (w * h) / 1_000_000
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("📏 Dimensions", f"{w}×{h}")
            with col_b:
                st.metric("📊 Megapixels", f"{megapixels:.1f}MP")
            with col_c:
                st.metric("📐 Aspect", f"{aspect_ratio:.2f}")
    
    with col2:
        st.markdown("### ⚙️ AI Model Settings")
        
        if uploaded_file:
            
            # Model type selection (like Meshy AI)
            model_type = st.selectbox(
                "🎯 Model Type",
                options=["point_cloud", "low", "medium", "high"],
                format_func=lambda x: {
                    "point_cloud": "🔮 Point Cloud (Fastest)",
                    "low": "🏔️ Low-Poly Mesh (Fast)", 
                    "medium": "🌋 Medium-Detail Mesh (Balanced)",
                    "high": "⛰️ High-Detail Mesh (Slow, Best Quality)"
                }[x],
                index=2
            )
            
            # Enhancement style
            enhancement = st.selectbox(
                "🎨 Enhancement Style", 
                options=["edge_enhanced", "smooth_terrain", "sharp_details", "artistic"],
                format_func=lambda x: {
                    "edge_enhanced": "⚡ Edge Enhanced (Recommended)",
                    "smooth_terrain": "🌊 Smooth Terrain", 
                    "sharp_details": "🔍 Sharp Details",
                    "artistic": "🎭 Artistic (Creative)"
                }[x]
            )
            
            # Density and height settings
            col_a, col_b = st.columns(2)
            
            with col_a:
                density = st.select_slider(
                    "📢 Model Density",
                    options=["preview", "low", "medium", "high", "ultra_high"],
                    value="medium",
                    format_func=lambda x: {
                        "preview": "👁️ Preview",
                        "low": "⚡ Low", 
                        "medium": "⚖️ Medium",
                        "high": "🎯 High",
                        "ultra_high": "💎 Ultra"
                    }[x]
                )
            
            with col_b:
                height_scale = st.slider(
                    "📏 Height Intensity", 
                    min_value=10,
                    max_value=200,
                    value=60,
                    help="Controls the dramatic effect of the 3D conversion"
                )
            
            # Generate button (Meshy AI style)
            generate_clicked = st.button(
                "🚀 Generate 3D Model",
                type="primary",
                use_container_width=True,
                help="Click to start AI-powered 3D model generation"
            )
            
            if generate_clicked:
                # Processing animation
                st.markdown("""
                <div class="processing-card">
                    <h3>🤖 AI Processing Your Image...</h3>
                    <p>Our advanced algorithms are analyzing your image and generating a high-quality 3D model</p>
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Simulate Meshy AI-like processing steps
                    steps = [
                        "🔍 Analyzing image structure...",
                        "🧠 AI depth estimation...", 
                        "🎨 Processing colors and textures...",
                        "🔧 Generating point cloud...",
                        "🗿 Building 3D mesh...",
                        "✨ Applying final enhancements..."
                    ]
                    
                    for i, step in enumerate(steps):
                        status_text.text(step)
                        progress_bar.progress((i + 1) / len(steps))
                        time.sleep(0.5)  # Simulate processing time
                    
                    # Actual conversion
                    model, points, colors_array, stats = create_advanced_3d_model(
                        image, height_scale, enhancement, model_type, density
                    )
                    
                    # Store in session state
                    st.session_state.model = model
                    st.session_state.points = points
                    st.session_state.colors = colors_array
                    st.session_state.stats = stats
                    st.session_state.filename = uploaded_file.name
                    st.session_state.model_type = stats['model_type']
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ 3D model generation complete!")
                    
                    # Auto-refresh to show results
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Generation failed: {str(e)}")
        else:
            st.info("👆 Upload an image and click 'Generate 3D Model' to start")
    
    # Results section (shown after generation)
    if hasattr(st.session_state, 'model') and st.session_state.model is not None:
        
        st.markdown("---")
        
        # Success banner
        st.markdown(f"""
        <div class="success-glow">
            <h2 style="margin:0; color: #00ff7f;">🎉 3D Model Generated Successfully!</h2>
            <p style="margin:0.5rem 0 0 0; color: rgba(255,255,255,0.9);">
                Created a {st.session_state.stats['model_type']} with {st.session_state.stats['total_points']:,} points
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced Results layout
        result_col1, result_col2 = st.columns([1, 2], gap="large")
        
        with result_col1:
            st.markdown("### 📊 Model Statistics")
            
            # Stats in a nice format
            stats = st.session_state.stats
            
            st.markdown(f"""
            <div class="feature-card">
                <strong>🎯 Model Details:</strong><br>
                🔷 Type: {stats['model_type'].title()}<br>
                🔢 Points: {stats['total_points']:,}<br>
                📏 Dimensions: {stats['processed_dimensions']}<br>
                📊 Height Range: {stats['height_range']}<br>
                🎨 Style: {stats['enhancement'].replace('_', ' ').title()}<br>
                💎 Quality: {stats['mesh_quality'].title()}
            </div>
            """, unsafe_allow_html=True)
            
            # Download section
            st.markdown("### 💾 Download Your Model")
            
            try:
                downloads = save_model_to_bytes(
                    st.session_state.model, 
                    st.session_state.model_type,
                    st.session_state.filename
                )
                
                filename_base = os.path.splitext(st.session_state.filename)[0]
                
                # PLY download
                st.download_button(
                    "⬇️ Download PLY (Universal)",
                    data=downloads['ply'],
                    file_name=f"{filename_base}_3d_model.ply",
                    mime="application/octet-stream",
                    use_container_width=True
                )
                
                # OBJ download if available
                if 'obj' in downloads:
                    st.download_button(
                        "⬇️ Download OBJ (Mesh)",
                        data=downloads['obj'],
                        file_name=f"{filename_base}_3d_model.obj", 
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                
                st.markdown("**🎯 Compatible Software:**")
                st.markdown("• Blender • MeshLab • Maya • 3ds Max • Unity • Unreal Engine")
                
            except Exception as e:
                st.error(f"Download preparation failed: {e}")
        
        with result_col2:
            st.markdown("### 🎮 Enhanced 3D Viewer")
            
            # Viewer controls
            st.markdown("""
            <div class="viewer-controls">
                <strong>🎛️ Viewer Controls</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Viewer options in columns for better layout
            viewer_col1, viewer_col2 = st.columns(2)
            
            with viewer_col1:
                view_mode = st.selectbox(
                    "👁️ View Mode",
                    options=["point_cloud", "surface", "wireframe"],
                    format_func=lambda x: {
                        "point_cloud": "🔮 Point Cloud",
                        "surface": "🌊 Surface",
                        "wireframe": "📐 Wireframe"
                    }[x],
                    key="view_mode"
                )
            
            with viewer_col2:
                color_mode = st.selectbox(
                    "🎨 Color Mode",
                    options=["original", "height", "terrain", "grayscale"],
                    format_func=lambda x: {
                        "original": "🖼️ Original",
                        "height": "📈 Height Map",
                        "terrain": "🗻 Terrain",
                        "grayscale": "⚫ Grayscale"
                    }[x],
                    key="color_mode"
                )
            
            # Create and display enhanced 3D visualization
            try:
                fig = create_enhanced_3d_visualization(
                    st.session_state.points, 
                    st.session_state.colors,
                    view_mode=view_mode,
                    color_mode=color_mode
                )
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                
                # Controls help
                st.markdown("""
                <div class="feature-card">
                    <strong>🎮 Interactive Controls:</strong><br>
                    🖱️ <strong>Rotate:</strong> Click and drag<br>
                    🔍 <strong>Zoom:</strong> Scroll wheel or pinch<br>
                    📱 <strong>Pan:</strong> Right-click and drag<br>
                    🏠 <strong>Reset:</strong> Double-click<br>
                    📋 <strong>Save View:</strong> Camera icon in toolbar
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"3D visualization failed: {e}")
        
        # Additional visualization section
        st.markdown("---")
        st.markdown("### 🔍 Detailed Analysis")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Side-by-Side Comparison", "📈 Statistics", "🛠️ Technical Info"])
        
        with tab1:
            st.markdown("**Compare original colors vs height-based visualization:**")
            try:
                comparison_fig = create_side_by_side_view(st.session_state.points, st.session_state.colors)
                st.plotly_chart(comparison_fig, use_container_width=True, theme="streamlit")
            except Exception as e:
                st.error(f"Comparison view failed: {e}")
        
        with tab2:
            # Detailed statistics
            points = st.session_state.points
            colors = st.session_state.colors
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("📍 Total Points", f"{len(points):,}")
                st.metric("📏 X Range", f"{points[:, 0].max() - points[:, 0].min():.1f}")
                st.metric("📐 Y Range", f"{points[:, 1].max() - points[:, 1].min():.1f}")
            
            with col_stat2:
                st.metric("⛰️ Height Range", f"{points[:, 2].max() - points[:, 2].min():.2f}")
                st.metric("📊 Mean Height", f"{np.mean(points[:, 2]):.2f}")
                st.metric("📈 Height Std", f"{np.std(points[:, 2]):.2f}")
            
            with col_stat3:
                st.metric("🎨 Color Diversity", f"{len(np.unique(colors.reshape(-1, 3), axis=0)):,}")
                st.metric("💾 Model Size", f"{len(points) * 6 * 4 / 1024:.1f} KB")  # Rough estimate
                st.metric("🔗 Aspect Ratio", f"{(points[:, 0].max() - points[:, 0].min()) / (points[:, 1].max() - points[:, 1].min()):.2f}")
            
            # Height distribution histogram
            st.markdown("**📊 Height Distribution:**")
            height_hist = px.histogram(
                x=points[:, 2],
                nbins=50,
                title="Distribution of Heights in 3D Model",
                labels={'x': 'Height', 'y': 'Frequency'}
            )
            height_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )
            st.plotly_chart(height_hist, use_container_width=True)
        
        with tab3:
            st.markdown("**🔧 Technical Processing Details:**")
            
            tech_info = f"""
            **Original Image Processing:**
            - Source dimensions: {st.session_state.stats['original_dimensions']}
            - Processed dimensions: {st.session_state.stats['processed_dimensions']}
            - Downsample factor: {st.session_state.stats['downsample_factor']}x
            
            **3D Conversion Details:**
            - Enhancement type: {st.session_state.stats['enhancement'].replace('_', ' ').title()}
            - Model quality: {st.session_state.stats['mesh_quality'].title()}
            - Density setting: {st.session_state.stats['density'].replace('_', ' ').title()}
            - Total vertices: {st.session_state.stats['total_points']:,}
            
            **Coordinate System:**
            - X-axis: Image width (left to right)
            - Y-axis: Image height (top to bottom)  
            - Z-axis: Brightness-based elevation
            - Height mapping: Linear (0-255 brightness → 0-{height_scale} units)
            
            **File Format Support:**
            - PLY: Point cloud data with colors
            - OBJ: Mesh data (if mesh generation successful)
            - Compatible with: Blender, MeshLab, CloudCompare, etc.
            """
            
            st.markdown(tech_info)
        
        # Next steps section
        st.markdown("---")
        st.markdown("### 🚀 Next Steps")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown("""
            <div class="feature-card">
                <h4>🎨 Further Editing</h4>
                <p>Import your model into Blender for advanced texturing, lighting, rigging, and animation. The PLY format preserves colors perfectly.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown("""
            <div class="feature-card">
                <h4>🖨️ 3D Printing</h4>
                <p>Clean up the mesh in MeshLab, check for manifold edges, and export as STL for 3D printing. Consider scaling and support structures.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_c:
            st.markdown("""
            <div class="feature-card">
                <h4>🎮 Game Development</h4>
                <p>Use as terrain, environment assets, or decorative props in Unity, Unreal Engine, or Godot. Optimize polygon count as needed.</p>
            </div>
            """, unsafe_allow_html=True)

# Enhanced Sidebar with more information
with st.sidebar:
    st.markdown("### 🔬 Advanced Options")
    
    show_advanced = st.checkbox("Show Advanced Settings")
    
    if show_advanced:
        st.markdown("**🎯 Processing Pipeline:**")
        st.markdown("1. Image preprocessing & enhancement")
        st.markdown("2. Depth map generation from brightness") 
        st.markdown("3. 3D point cloud creation")
        st.markdown("4. Mesh reconstruction (optional)")
        st.markdown("5. Quality optimization & cleanup")
        
        st.markdown("**💡 Pro Tips:**")
        st.markdown("• High contrast images = better 3D structure")
        st.markdown("• Portraits and landscapes work excellently") 
        st.markdown("• Try different enhancement styles for variety")
        st.markdown("• Start with medium quality, then increase")
        st.markdown("• Use wireframe view to see mesh structure")
        
        st.markdown("**🎨 View Modes Explained:**")
        st.markdown("• **Point Cloud**: Raw 3D points, fastest rendering")
        st.markdown("• **Surface**: Smooth interpolated surface")
        st.markdown("• **Wireframe**: Shows mesh structure and topology")
        
        st.markdown("**🌈 Color Modes:**")
        st.markdown("• **Original**: Preserves image colors")
        st.markdown("• **Height**: Colors based on elevation")
        st.markdown("• **Terrain**: Earth-like color mapping")
        st.markdown("• **Grayscale**: Monochrome visualization")
    
    st.markdown("---")
    st.markdown("**🤖 Powered by:**")
    st.markdown("• Open3D for 3D processing")
    st.markdown("• OpenCV for image analysis") 
    st.markdown("• Plotly for interactive visualization")
    st.markdown("• SciPy for advanced algorithms")
    
    # Show current session info if model exists
    if hasattr(st.session_state, 'model') and st.session_state.model is not None:
        st.markdown("---")
        st.markdown("**📊 Current Session:**")
        st.markdown(f"• Model: {st.session_state.stats['model_type']}")
        st.markdown(f"• Points: {st.session_state.stats['total_points']:,}")
        st.markdown(f"• File: {st.session_state.filename}")

if __name__ == "__main__":
    main()