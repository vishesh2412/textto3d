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
    initial_sidebar_state="expanded"
)

# Simplified CSS with red, black, and blue theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    
    .stApp {
        background: #000000;
        color: #ffffff;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(90deg, rgba(220,20,60,0.1) 0%, rgba(0,0,0,0.9) 50%, rgba(65,105,225,0.1) 100%);
        border: 1px solid rgba(220,20,60,0.3);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .header-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #dc143c, #4169e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .header-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: rgba(255,255,255,0.8);
        margin: 0.5rem 0 0 0;
    }
    
    /* Card styling */
    .main-card {
        background: rgba(10,10,10,0.8);
        border: 1px solid rgba(220,20,60,0.3);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .result-card {
        background: rgba(10,10,10,0.8);
        border: 1px solid rgba(65,105,225,0.3);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .success-banner {
        background: linear-gradient(90deg, rgba(34,139,34,0.2) 0%, rgba(65,105,225,0.2) 100%);
        border: 1px solid #228b22;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .processing-banner {
        background: linear-gradient(90deg, rgba(255,140,0,0.2) 0%, rgba(220,20,60,0.2) 100%);
        border: 1px solid #ff8c00;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Interactive elements */
    .stSelectbox > div > div, .stSlider > div > div {
        background: rgba(10,10,10,0.8) !important;
        border: 1px solid rgba(220,20,60,0.3) !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #dc143c, #4169e1) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(220,20,60,0.3) !important;
    }
    
    /* File uploader */
    .stFileUploader > section {
        background: rgba(10,10,10,0.8) !important;
        border: 2px dashed rgba(220,20,60,0.4) !important;
        border-radius: 10px !important;
    }
    
    /* Metrics */
    .metric-card {
        background: rgba(10,10,10,0.6);
        border: 1px solid rgba(65,105,225,0.3);
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(10,10,10,0.8) !important;
        border-radius: 8px !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border: 1px solid rgba(220,20,60,0.2) !important;
        color: white !important;
        border-radius: 6px !important;
        margin: 0.1rem !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(45deg, rgba(220,20,60,0.3), rgba(65,105,225,0.3)) !important;
        border-color: #dc143c !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #dc143c, #4169e1) !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(10,10,10,0.95) 0%, rgba(26,26,46,0.95) 100%) !important;
        border-right: 1px solid rgba(220,20,60,0.3) !important;
    }
    
    /* Text styling */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    p, span, div {
        color: rgba(255,255,255,0.9) !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.2);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #dc143c, #4169e1);
        border-radius: 3px;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .header-title { font-size: 2rem; }
        .header-subtitle { font-size: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

def preprocess_image_for_3d(image, enhancement_type="edge_enhanced"):
    """Advanced image preprocessing for better 3D reconstruction"""
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
        edges = cv2.Canny(gray, 50, 150)
        gray = cv2.addWeighted(gray, 0.8, edges, 0.2, 0)
    elif enhancement_type == "smooth_terrain":
        gray = gaussian_filter(gray, sigma=1.5)
    elif enhancement_type == "sharp_details":
        gray = cv2.equalizeHist(gray)
    elif enhancement_type == "artistic":
        gray = np.power(gray / 255.0, 0.7) * 255
        gray = gray.astype(np.uint8)
    
    return gray, color

def generate_mesh_from_pointcloud(points, colors, mesh_quality="medium"):
    """Generate a mesh from point cloud using different algorithms"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    if mesh_quality == "point_cloud":
        return pcd, "point_cloud"
    
    try:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
        
        if mesh_quality == "low":
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6)
        elif mesh_quality == "medium": 
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        else:  # high quality
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
        
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        return mesh, "mesh"
    except Exception as e:
        st.warning(f"Mesh generation failed: {e}. Returning point cloud instead.")
        return pcd, "point_cloud"

def create_advanced_3d_model(image, height_scale, enhancement_type, mesh_quality, density_factor):
    """Create advanced 3D model with various options"""
    gray, color = preprocess_image_for_3d(image, enhancement_type)
    
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
    
    if downsample > 1:
        gray = gray[::downsample, ::downsample]
        color = color[::downsample, ::downsample]
    
    height, width = gray.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    z_coords = gray.astype(np.float32) / 255.0 * height_scale
    
    if enhancement_type == "artistic":
        noise = np.random.normal(0, height_scale * 0.02, z_coords.shape)
        z_coords += noise
    
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten() 
    z_flat = z_coords.flatten()
    
    points = np.column_stack((x_flat, y_flat, z_flat))
    colors_flat = color.reshape(-1, 3) / 255.0
    
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

def generate_ply_content(points, colors):
    """Generate PLY file content as string"""
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    vertices = []
    for i, (point, color) in enumerate(zip(points, colors)):
        r, g, b = [int(c * 255) for c in color]
        vertices.append(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {r} {g} {b}")
    
    return header + "\n".join(vertices)

def create_ply_viewer_html(ply_content, height=400):
    """Create HTML with Three.js PLY viewer"""
    
    # Encode PLY content for embedding
    import base64
    ply_b64 = base64.b64encode(ply_content.encode()).decode()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: #000;
                overflow: hidden;
                font-family: Arial, sans-serif;
            }}
            #container {{
                width: 100%;
                height: {height}px;
                position: relative;
                border-radius: 10px;
                overflow: hidden;
            }}
            #info {{
                position: absolute;
                top: 10px;
                left: 10px;
                color: #fff;
                font-size: 12px;
                z-index: 100;
                background: rgba(0,0,0,0.7);
                padding: 8px;
                border-radius: 5px;
                border: 1px solid #dc143c;
            }}
            #controls {{
                position: absolute;
                bottom: 10px;
                left: 10px;
                color: #fff;
                font-size: 11px;
                z-index: 100;
                background: rgba(0,0,0,0.7);
                padding: 8px;
                border-radius: 5px;
                border: 1px solid #4169e1;
            }}
        </style>
    </head>
    <body>
        <div id="container">
            <div id="info">🎯 PLY 3D Model Viewer</div>
            <div id="controls">
                🖱️ Rotate: Left Click + Drag<br>
                🔍 Zoom: Mouse Wheel<br>
                📱 Pan: Right Click + Drag
            </div>
        </div>
        
        <script>
            // Decode PLY content
            const plyContent = atob('{ply_b64}');
            
            let scene, camera, renderer, controls, pointCloud;
            let mouseX = 0, mouseY = 0;
            let isMouseDown = false, isRightClick = false;
            
            function init() {{
                // Scene setup
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0x000000);
                
                // Camera setup
                camera = new THREE.PerspectiveCamera(75, window.innerWidth / {height}, 0.1, 1000);
                camera.position.set(50, 50, 100);
                
                // Renderer setup
                const container = document.getElementById('container');
                renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(container.offsetWidth, {height});
                container.appendChild(renderer.domElement);
                
                // Parse PLY content
                const geometry = parsePLY(plyContent);
                
                // Create point cloud material
                const material = new THREE.PointsMaterial({{
                    vertexColors: true,
                    size: 2,
                    sizeAttenuation: true
                }});
                
                // Create point cloud
                pointCloud = new THREE.Points(geometry, material);
                scene.add(pointCloud);
                
                // Center the model
                const box = new THREE.Box3().setFromObject(pointCloud);
                const center = box.getCenter(new THREE.Vector3());
                pointCloud.position.sub(center);
                
                // Adjust camera distance based on model size
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                camera.position.set(maxDim * 1.2, maxDim * 1.2, maxDim * 1.2);
                camera.lookAt(0, 0, 0);
                
                // Add lighting
                const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
                scene.add(ambientLight);
                
                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                directionalLight.position.set(50, 50, 50);
                scene.add(directionalLight);
                
                // Mouse controls
                setupControls();
                
                // Start animation
                animate();
            }}
            
            function parsePLY(content) {{
                const lines = content.split('\\n');
                let vertexCount = 0;
                let headerEnded = false;
                let vertices = [];
                let colors = [];
                
                for (let i = 0; i < lines.length; i++) {{
                    const line = lines[i].trim();
                    
                    if (!headerEnded) {{
                        if (line.startsWith('element vertex')) {{
                            vertexCount = parseInt(line.split(' ')[2]);
                        }}
                        if (line === 'end_header') {{
                            headerEnded = true;
                            continue;
                        }}
                    }} else {{
                        if (vertices.length < vertexCount && line) {{
                            const parts = line.split(' ');
                            if (parts.length >= 6) {{
                                // Position
                                vertices.push(
                                    parseFloat(parts[0]),
                                    parseFloat(parts[1]),
                                    parseFloat(parts[2])
                                );
                                
                                // Color
                                colors.push(
                                    parseInt(parts[3]) / 255,
                                    parseInt(parts[4]) / 255,
                                    parseInt(parts[5]) / 255
                                );
                            }}
                        }}
                    }}
                }}
                
                const geometry = new THREE.BufferGeometry();
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
                geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
                
                return geometry;
            }}
            
            function setupControls() {{
                const canvas = renderer.domElement;
                
                canvas.addEventListener('mousedown', (event) => {{
                    isMouseDown = true;
                    isRightClick = event.button === 2;
                    mouseX = event.clientX;
                    mouseY = event.clientY;
                }});
                
                canvas.addEventListener('mouseup', () => {{
                    isMouseDown = false;
                }});
                
                canvas.addEventListener('mousemove', (event) => {{
                    if (!isMouseDown) return;
                    
                    const deltaX = event.clientX - mouseX;
                    const deltaY = event.clientY - mouseY;
                    
                    if (isRightClick) {{
                        // Pan
                        camera.position.x -= deltaX * 0.1;
                        camera.position.y += deltaY * 0.1;
                    }} else {{
                        // Rotate
                        pointCloud.rotation.y += deltaX * 0.01;
                        pointCloud.rotation.x += deltaY * 0.01;
                    }}
                    
                    mouseX = event.clientX;
                    mouseY = event.clientY;
                }});
                
                canvas.addEventListener('wheel', (event) => {{
                    event.preventDefault();
                    const scale = event.deltaY > 0 ? 1.1 : 0.9;
                    camera.position.multiplyScalar(scale);
                }});
                
                canvas.addEventListener('contextmenu', (event) => {{
                    event.preventDefault();
                }});
            }}
            
            function animate() {{
                requestAnimationFrame(animate);
                
                // Auto rotation when not interacting
                if (!isMouseDown) {{
                    pointCloud.rotation.y += 0.005;
                }}
                
                renderer.render(scene, camera);
            }}
            
            // Handle window resize
            window.addEventListener('resize', () => {{
                const container = document.getElementById('container');
                camera.aspect = container.offsetWidth / {height};
                camera.updateProjectionMatrix();
                renderer.setSize(container.offsetWidth, {height});
            }});
            
            // Initialize
            init();
        </script>
    </body>
    </html>
    """
    
    return html_content

def create_3d_visualization(points, colors, view_mode="point_cloud", color_mode="original"):
    """Create interactive 3D visualization"""
    
    max_points = 6000 if view_mode == "point_cloud" else 4000
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
        z_norm = (points_sample[:, 2] - points_sample[:, 2].min()) / (points_sample[:, 2].max() - points_sample[:, 2].min())
        colors_rgb = px.colors.sample_colorscale('plasma', z_norm)
        colors_rgb = np.array([[int(c[4:6], 16)/255, int(c[6:8], 16)/255, int(c[8:10], 16)/255] for c in colors_rgb])
    elif color_mode == "themed":
        z_norm = (points_sample[:, 2] - points_sample[:, 2].min()) / (points_sample[:, 2].max() - points_sample[:, 2].min())
        # Create custom red-blue colorscale
        colors_rgb = np.zeros((len(z_norm), 3))
        colors_rgb[:, 0] = 1 - z_norm  # Red decreases with height
        colors_rgb[:, 2] = z_norm      # Blue increases with height
    else:  # grayscale
        gray_vals = np.mean(colors_sample, axis=1)
        colors_rgb = np.column_stack([gray_vals, gray_vals, gray_vals])
    
    colors_plotly = ['rgb({},{},{})'.format(
        int(c[0]*255), int(c[1]*255), int(c[2]*255)
    ) for c in colors_rgb]
    
    fig = go.Figure()
    
    if view_mode == "point_cloud":
        fig.add_trace(go.Scatter3d(
            x=points_sample[:, 0],
            y=points_sample[:, 1], 
            z=points_sample[:, 2],
            mode='markers',
            marker=dict(
                size=2.5,
                color=colors_plotly,
                opacity=0.8,
                line=dict(width=0)
            ),
            name='Points',
            hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
        ))
    
    elif view_mode == "surface":
        try:
            from scipy.interpolate import griddata
            
            if len(points_sample) > 1500:
                indices = np.random.choice(len(points_sample), 1500, replace=False)
                points_surf = points_sample[indices]
            else:
                points_surf = points_sample
            
            xi = np.linspace(points_surf[:, 0].min(), points_surf[:, 0].max(), 40)
            yi = np.linspace(points_surf[:, 1].min(), points_surf[:, 1].max(), 40)
            xi_grid, yi_grid = np.meshgrid(xi, yi)
            
            zi_grid = griddata((points_surf[:, 0], points_surf[:, 1]), points_surf[:, 2], 
                             (xi_grid, yi_grid), method='linear', fill_value=0)
            
            fig.add_trace(go.Surface(
                x=xi_grid, y=yi_grid, z=zi_grid,
                colorscale='plasma',
                showscale=True,
                name='Surface'
            ))
        except Exception:
            fig.add_trace(go.Scatter3d(
                x=points_sample[:, 0], y=points_sample[:, 1], z=points_sample[:, 2],
                mode='markers', marker=dict(size=2.5, color=colors_plotly, opacity=0.8),
                name='Points'
            ))
    
    elif view_mode == "wireframe":
        try:
            if len(points_sample) > 800:
                indices = np.random.choice(len(points_sample), 800, replace=False)
                points_wire = points_sample[indices]
            else:
                points_wire = points_sample
            
            points_2d = points_wire[:, :2]
            tri = Delaunay(points_2d)
            
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
                line=dict(color='#dc143c', width=1.5),
                name='Wireframe',
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter3d(
                x=points_wire[:, 0], y=points_wire[:, 1], z=points_wire[:, 2],
                mode='markers',
                marker=dict(size=3, color='#4169e1', opacity=1),
                name='Vertices'
            ))
        except Exception:
            fig.add_trace(go.Scatter3d(
                x=points_sample[:, 0], y=points_sample[:, 1], z=points_sample[:, 2],
                mode='markers', marker=dict(size=2.5, color=colors_plotly, opacity=0.8),
                name='Points'
            ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"3D Model - {view_mode.replace('_', ' ').title()}",
            font=dict(size=18, color="#ffffff"),
            x=0.5
        ),
        scene=dict(
            xaxis=dict(
                title="X Axis",
                backgroundcolor="rgba(0,0,0,0.8)",
                gridcolor="rgba(220,20,60,0.3)"
            ),
            yaxis=dict(
                title="Y Axis",
                backgroundcolor="rgba(0,0,0,0.8)",
                gridcolor="rgba(65,105,225,0.3)"
            ),
            zaxis=dict(
                title="Z Axis",
                backgroundcolor="rgba(0,0,0,0.8)",
                gridcolor="rgba(255,255,255,0.2)"
            ),
            bgcolor="rgba(0,0,0,1)",
            camera=dict(
                eye=dict(x=1.3, y=1.3, z=1.1)
            ),
            aspectmode='cube'
        ),
        paper_bgcolor="rgba(0,0,0,1)",
        plot_bgcolor="rgba(0,0,0,1)",
        font=dict(color="#ffffff"),
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        height=400
    )
    
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
    
    if model_type == "mesh":
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp_file:
            try:
                o3d.io.write_triangle_mesh(tmp_file.name, model)
                with open(tmp_file.name, 'rb') as f:
                    downloads['obj'] = f.read()
            except:
                pass
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    return downloads

def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <div class="header-title">🚀 AI 3D Model Generator</div>
        <div class="header-subtitle">Transform Images into Interactive 3D Models</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main layout - two columns
    col1, col2 = st.columns([1, 1.2], gap="medium")
    
    with col1:
        st.markdown("""
        <div class="main-card">
            <h4 style="color: #dc143c; margin-top: 0;">📁 Upload Image</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
            help="Supported: PNG, JPG, JPEG, BMP, TIFF, WebP"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            
            # Image preview
            st.image(image, caption=f"📷 {uploaded_file.name}", use_column_width=True)
            
            # Image info
            w, h = image.size
            megapixels = (w * h) / 1_000_000
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class="metric-card">
                    <strong style="color: #4169e1;">📐 Size</strong><br>
                    {w}×{h}
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""
                <div class="metric-card">
                    <strong style="color: #dc143c;">⚡ Quality</strong><br>
                    {megapixels:.1f}MP
                </div>
                """, unsafe_allow_html=True)
            
            # Settings
            st.markdown("""
            <div class="main-card">
                <h4 style="color: #4169e1; margin-top: 0;">⚙️ Settings</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Model quality
            model_type = st.selectbox(
                "🎯 Model Quality",
                options=["point_cloud", "low", "medium", "high"],
                format_func=lambda x: {
                    "point_cloud": "📍 Point Cloud (Fast)",
                    "low": "⚡ Low Quality (Fast)", 
                    "medium": "🎯 Medium Quality (Balanced)",
                    "high": "🚀 High Quality (Detailed)"
                }[x],
                index=2
            )
            
            # Enhancement
            enhancement = st.selectbox(
                "🎨 Enhancement", 
                options=["edge_enhanced", "smooth_terrain", "sharp_details", "artistic"],
                format_func=lambda x: {
                    "edge_enhanced": "⚡ Edge Enhanced",
                    "smooth_terrain": "🌊 Smooth Surface", 
                    "sharp_details": "🔍 Sharp Details",
                    "artistic": "🎭 Artistic Style"
                }[x]
            )
            
            # Advanced settings in expandable section
            with st.expander("Advanced Settings"):
                density = st.select_slider(
                    "Point Density",
                    options=["preview", "low", "medium", "high", "ultra_high"],
                    value="medium"
                )
                
                height_scale = st.slider(
                    "Height Scale", 
                    min_value=10, max_value=200, value=60
                )
            
            # Generate button
            generate_clicked = st.button(
                "🚀 Generate 3D Model",
                type="primary",
                use_container_width=True
            )
            
            if generate_clicked:
                # Processing
                st.markdown("""
                <div class="processing-banner">
                    <h4 style="color: #ff8c00; margin: 0;">🔄 Processing...</h4>
                    <p style="margin: 0.5rem 0 0 0;">Generating 3D model from your image</p>
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    steps = [
                        "🔍 Analyzing image...",
                        "🧮 Calculating depth...", 
                        "🎨 Processing colors...",
                        "📊 Building point cloud...",
                        "🔧 Creating mesh...",
                        "✨ Finalizing..."
                    ]
                    
                    for i, step in enumerate(steps):
                        status_text.text(step)
                        progress_bar.progress((i + 1) / len(steps))
                        time.sleep(0.5)
                    
                    # Generate model
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
                    status_text.text("✅ Model generated successfully!")
                    
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.info("👆 Upload an image to get started")
    
    with col2:
        if hasattr(st.session_state, 'model') and st.session_state.model is not None:
            # Success message
            st.markdown(f"""
            <div class="success-banner">
                <h4 style="color: #228b22; margin: 0;">🎉 Success!</h4>
                <p style="margin: 0.5rem 0 0 0;">Generated {st.session_state.stats['model_type']} with {st.session_state.stats['total_points']:,} points</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Viewer controls
            st.markdown("""
            <div class="result-card">
                <h4 style="color: #4169e1; margin-top: 0;">👁️ 3D Viewer</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Controls in two columns
            ctrl_col1, ctrl_col2 = st.columns(2)
            
            with ctrl_col1:
                view_mode = st.selectbox(
                    "View Mode",
                    options=["point_cloud", "surface", "wireframe"],
                    format_func=lambda x: {
                        "point_cloud": "📍 Point Cloud",
                        "surface": "🌊 Surface",
                        "wireframe": "🔗 Wireframe"
                    }[x],
                    key="view_mode"
                )
            
            with ctrl_col2:
                color_mode = st.selectbox(
                    "Color Mode",
                    options=["original", "height", "themed", "grayscale"],
                    format_func=lambda x: {
                        "original": "🖼️ Original",
                        "height": "📈 Height Map",
                        "themed": "🎨 Red-Blue Theme",
                        "grayscale": "⚫ Grayscale"
                    }[x],
                    key="color_mode"
                )
            
            # 3D Visualization options
            viz_tabs = st.tabs(["📊 Plotly Viewer", "🎯 PLY Viewer"])
            
            with viz_tabs[0]:
                # Original Plotly visualization
                try:
                    fig = create_3d_visualization(
                        st.session_state.points, 
                        st.session_state.colors,
                        view_mode=view_mode,
                        color_mode=color_mode
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Plotly visualization error: {e}")
            
            with viz_tabs[1]:
                # PLY viewer with Three.js
                try:
                    # Generate PLY content
                    ply_content = generate_ply_content(st.session_state.points, st.session_state.colors)
                    
                    # Create and display PLY viewer
                    ply_viewer_html = create_ply_viewer_html(ply_content, height=400)
                    
                    st.markdown("**🎯 Interactive PLY Model Viewer**")
                    st.components.v1.html(ply_viewer_html, height=400)
                    
                    st.markdown("""
                    <div style="background: rgba(65,105,225,0.1); padding: 0.8rem; 
                                border-radius: 8px; border: 1px solid rgba(65,105,225,0.3); 
                                font-size: 0.85rem; margin-top: 0.5rem;">
                        💡 <strong>PLY Viewer Features:</strong><br>
                        • Real-time 3D rendering with Three.js<br>
                        • Interactive mouse controls (rotate, zoom, pan)<br>
                        • Full color preservation from original image<br>
                        • Auto-rotation when not interacting
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"PLY viewer error: {e}")
                    st.info("Fallback: Use the Plotly viewer above or download the PLY file.")
            
            # Stats and downloads
            stats_col, download_col = st.columns(2)
            
            with stats_col:
                st.markdown("""
                <div class="result-card">
                    <h4 style="color: #dc143c; margin-top: 0;">📊 Statistics</h4>
                </div>
                """, unsafe_allow_html=True)
                
                stats = st.session_state.stats
                st.markdown(f"""
                <div style="font-size: 0.9rem;">
                    <p><strong>Type:</strong> {stats['model_type'].title()}</p>
                    <p><strong>Points:</strong> {stats['total_points']:,}</p>
                    <p><strong>Size:</strong> {stats['processed_dimensions']}</p>
                    <p><strong>Z-Range:</strong> {stats['height_range']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with download_col:
                st.markdown("""
                <div class="result-card">
                    <h4 style="color: #228b22; margin-top: 0;">💾 Download</h4>
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    downloads = save_model_to_bytes(
                        st.session_state.model, 
                        st.session_state.model_type,
                        st.session_state.filename
                    )
                    
                    filename_base = os.path.splitext(st.session_state.filename)[0]
                    
                    # Downloads
                    st.download_button(
                        "📁 Download PLY",
                        data=downloads['ply'],
                        file_name=f"{filename_base}_3d_model.ply",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                    
                    if 'obj' in downloads:
                        st.download_button(
                            "📁 Download OBJ",
                            data=downloads['obj'],
                            file_name=f"{filename_base}_3d_mesh.obj", 
                            mime="application/octet-stream",
                            use_container_width=True
                        )
                    
                    st.markdown("""
                    <div style="background: rgba(65,105,225,0.1); padding: 0.8rem; 
                                border-radius: 8px; border: 1px solid rgba(65,105,225,0.3); 
                                font-size: 0.8rem; margin-top: 0.5rem;">
                        <strong>Compatible with:</strong><br>
                        Blender • MeshLab • Unity • Maya
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Download preparation failed: {e}")
            
            # Additional analysis in tabs
            st.markdown("---")
            
            tab1, tab2 = st.tabs(["📈 Analysis", "🛠️ Technical"])
            
            with tab1:
                # Quick analysis
                points = st.session_state.points
                colors = st.session_state.colors
                
                anal_col1, anal_col2, anal_col3 = st.columns(3)
                
                with anal_col1:
                    st.metric("Points", f"{len(points):,}")
                
                with anal_col2:
                    st.metric("Max Height", f"{points[:, 2].max():.2f}")
                
                with anal_col3:
                    st.metric("Min Height", f"{points[:, 2].min():.2f}")
                
                # Height distribution chart
                if len(points) > 0:
                    height_hist = px.histogram(
                        x=points[:, 2],
                        nbins=30,
                        title="Height Distribution",
                        color_discrete_sequence=['#dc143c']
                    )
                    height_hist.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#ffffff"),
                        height=250
                    )
                    st.plotly_chart(height_hist, use_container_width=True)
            
            with tab2:
                # Technical details
                st.markdown(f"""
                **Processing Details:**
                - Original: {st.session_state.stats['original_dimensions']}
                - Processed: {st.session_state.stats['processed_dimensions']}
                - Enhancement: {st.session_state.stats['enhancement'].replace('_', ' ').title()}
                - Quality: {st.session_state.stats['mesh_quality'].title()}
                - Density: {st.session_state.stats['density'].replace('_', ' ').title()}
                
                **File Formats:**
                - PLY: Point cloud with colors
                - OBJ: Mesh data (if available)
                
                **Usage:**
                Import into 3D software for further editing, 3D printing, or game development.
                """)
                
        else:
            st.markdown("""
            <div class="result-card" style="text-align: center; padding: 2rem;">
                <h4 style="color: #4169e1;">🎯 Ready to Create</h4>
                <p>Upload an image and configure settings to generate your 3D model</p>
                <div style="margin: 1rem 0;">
                    <span style="color: #dc143c;">📷</span> → 
                    <span style="color: #4169e1;">🔄</span> → 
                    <span style="color: #228b22;">📐</span>
                </div>
                <p style="font-size: 0.9rem; color: rgba(255,255,255,0.7);">
                    Image → Processing → 3D Model
                </p>
            </div>
            """, unsafe_allow_html=True)

# Sidebar with tips and info
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(180deg, rgba(220,20,60,0.1) 0%, rgba(65,105,225,0.1) 100%); 
                padding: 1rem; border-radius: 10px; border: 1px solid rgba(220,20,60,0.3); margin-bottom: 1rem;">
        <h4 style="color: #dc143c; text-align: center; margin: 0;">🧠 AI Engine</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 💡 Tips for Best Results")
    
    tips = [
        "📷 **High Contrast**: Images with clear light/dark areas work best",
        "🎯 **Clear Details**: Sharp, focused images produce better models",
        "⚡ **Start Simple**: Try medium quality first, then increase",
        "🎨 **Experiment**: Different enhancements create unique effects",
        "📐 **Size Matters**: Larger images = more detail (but slower processing)"
    ]
    
    for tip in tips:
        st.markdown(f"- {tip}")
    
    st.markdown("---")
    
    st.markdown("### 🛠️ Enhancement Modes")
    
    enhancements = {
        "⚡ Edge Enhanced": "Emphasizes edges and boundaries",
        "🌊 Smooth Surface": "Creates smooth, flowing surfaces", 
        "🔍 Sharp Details": "Enhances contrast and clarity",
        "🎭 Artistic Style": "Adds creative noise and variation"
    }
    
    for name, desc in enhancements.items():
        st.markdown(f"**{name}**")
        st.markdown(f"*{desc}*")
        st.markdown("")
    
    st.markdown("---")
    
    st.markdown("### 📊 Model Quality")
    
    qualities = {
        "📍 Point Cloud": "Fast preview, no mesh",
        "⚡ Low Quality": "Quick processing, basic mesh", 
        "🎯 Medium Quality": "Balanced speed and detail",
        "🚀 High Quality": "Detailed mesh, slower processing"
    }
    
    for name, desc in qualities.items():
        st.markdown(f"**{name}**")
        st.markdown(f"*{desc}*")
        st.markdown("")
    
    # Show current session info if model exists
    if hasattr(st.session_state, 'model') and st.session_state.model is not None:
        st.markdown("---")
        st.markdown("### 📋 Current Session")
        st.markdown(f"""
        **File:** {st.session_state.filename}
        
        **Model:** {st.session_state.stats['model_type'].title()}
        
        **Points:** {st.session_state.stats['total_points']:,}
        
        **Status:** ✅ Ready for download
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; 
            background: linear-gradient(90deg, rgba(220,20,60,0.1) 0%, rgba(0,0,0,0.8) 50%, rgba(65,105,225,0.1) 100%); 
            border-radius: 10px; border: 1px solid rgba(220,20,60,0.2);">
    <h4 style="color: #dc143c; margin: 0;">🚀 AI 3D Model Generator</h4>
    <p style="color: rgba(255,255,255,0.7); margin: 0.5rem 0 0 0; font-size: 0.9rem;">
        Powered by Open3D • OpenCV • Plotly • SciPy
    </p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
