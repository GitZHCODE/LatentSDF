"""
PathSelect - Standalone Latent SDF Navigator
Loads pre-trained auto-decoder models and allows interactive path creation in latent space.
"""

import sys
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import pygame

# # Set SDL environment variables before importing pygame
# os.environ['SDL_VIDEODRIVER'] = 'windows'
# os.environ['SDL_AUDIODRIVER'] = 'directsound'
# os.environ['SDL_FORCE_SDL2'] = '1'

# # Block SDL3 imports
# class SDL3ImportBlocker:
#     def find_spec(self, name, path, target=None):
#         if 'SDL3' in name or 'sdl3' in name.lower():
#             print(f"🚫 Blocking SDL3 import: {name}")
#             return None
#         return None

# sys.meta_path.insert(0, SDL3ImportBlocker())

# # Now import pygame
# try:
#     import pygame
#     print(f"✅ Pygame imported - Version: {pygame.version.ver}")
#     print(f"✅ SDL version: {pygame.version.SDL}")
    
#     # Initialize pygame with error handling
#     pygame.init()
#     print("✅ Pygame initialized successfully")
    
#     # Verify we're using SDL2 (convert to string for comparison)
#     sdl_version_str = str(pygame.version.SDL)
#     if sdl_version_str.startswith('2.'):
#         print("✅ Using SDL2 as expected")
#     else:
#         print(f"⚠️  Warning: Unexpected SDL version: {sdl_version_str}")
        
# except ImportError as e:
#     print(f"❌ Failed to import pygame: {e}")
#     print("Please install pygame: pip install pygame")
#     sys.exit(1)
# except Exception as e:
#     print(f"❌ Failed to initialize pygame: {e}")
#     print("This may be due to SDL library conflicts.")
#     print("Try the following:")
#     print("1. Install Visual C++ Redistributable")
#     print("2. Install DirectX runtime")
#     print("3. Ensure only SDL2 libraries are present")
    
#     # Try to get more specific error information
#     try:
#         import pygame.version
#         print(f"Pygame version: {pygame.version.ver}")
#         print(f"SDL version: {pygame.version.SDL}")
#     except:
#         pass
    
#     sys.exit(1)

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
LATENT_VIEW_WIDTH = 600
LATENT_VIEW_HEIGHT = 600
SDF_VIEW_WIDTH = 400
SDF_VIEW_HEIGHT = 400

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
YELLOW = (255, 255, 0)

class ModelLoader:
    """Handles loading of pre-trained model components"""
    
    def __init__(self, models_folder="models"):
        # Make the models folder path absolute to avoid directory issues
        if not os.path.isabs(models_folder):
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.models_folder = os.path.join(script_dir, models_folder)
        else:
            self.models_folder = models_folder
        self.decoder = None
        self.latent_codes = None
        self.coords_flat = None
        self.data_info = None
        
    def check_model_files(self):
        """Check if all required model files exist"""
        required_files = [
            "decoder_model.h5",
            "latent_codes.npy",
            "coords_flat.npy",
            "data.json"
        ]
        
        print(f"🔍 Checking model files in: {self.models_folder}")
        print(f"🔍 Models folder exists: {os.path.exists(self.models_folder)}")
        
        missing_files = []
        for filename in required_files:
            filepath = os.path.join(self.models_folder, filename)
            file_exists = os.path.exists(filepath)
            print(f"   {'✅' if file_exists else '❌'} {filename}: {filepath}")
            if not file_exists:
                missing_files.append(filename)
        
        return missing_files
    
    def load_model(self):
        """Load all model components"""
        try:
            print(f"Loading model from {self.models_folder}...")
            
            # Check files first
            missing = self.check_model_files()
            if missing:
                print(f"❌ Missing model files: {missing}")
                return False
            
            # Load decoder model with TensorFlow compatibility handling
            decoder_path = os.path.join(self.models_folder, "decoder_model.h5")
            self.decoder = self._load_model_with_compatibility(decoder_path)
            print(f"✅ Loaded decoder model")
            
            # Load latent codes
            latent_path = os.path.join(self.models_folder, "latent_codes.npy")
            self.latent_codes = np.load(latent_path)
            print(f"✅ Loaded latent codes: {self.latent_codes.shape}")
            
            # Load coordinates
            coords_path = os.path.join(self.models_folder, "coords_flat.npy")
            self.coords_flat = np.load(coords_path)
            print(f"✅ Loaded coordinates: {self.coords_flat.shape}")
            
            # Load data info (contains all metadata)
            data_info_path = os.path.join(self.models_folder, "data.json")
            with open(data_info_path, 'r') as f:
                self.data_info = json.load(f)
            
            # Extract grid dimensions from data.json
            # Handle both array and direct value formats
            x_count_val = self.data_info['scalar_field_XCount']
            y_count_val = self.data_info['scalar_field_YCount']
            
            if isinstance(x_count_val, list):
                self.x_count = int(x_count_val[0])
            else:
                self.x_count = int(x_count_val)
                
            if isinstance(y_count_val, list):
                self.y_count = int(y_count_val[0])
            else:
                self.y_count = int(y_count_val)
            
            # Get model metadata if available
            self.model_metadata = self.data_info.get('model_metadata', {})
            
            # Load latent range information (for new natural range models)
            self.latent_range_info = self.model_metadata.get('latent_range', {})
            self.normalization_type = self.latent_range_info.get('normalization_type', 'tanh')
            
            if self.normalization_type == 'natural_range':
                self.latent_min = np.array(self.latent_range_info['exploration_min'])
                self.latent_max = np.array(self.latent_range_info['exploration_max'])
                self.latent_min_natural = np.array(self.latent_range_info['natural_min'])
                self.latent_max_natural = np.array(self.latent_range_info['natural_max'])
                print(f"✅ Using natural range normalization")
                print(f"   Natural range: [{self.latent_min_natural[0]:.3f}, {self.latent_min_natural[1]:.3f}] to [{self.latent_max_natural[0]:.3f}, {self.latent_max_natural[1]:.3f}]")
                print(f"   Exploration range: [{self.latent_min[0]:.3f}, {self.latent_min[1]:.3f}] to [{self.latent_max[0]:.3f}, {self.latent_max[1]:.3f}]")
            else:
                # Backward compatibility: use tanh normalization ([-1, 1])
                self.latent_min = np.array([-1.0, -1.0])
                self.latent_max = np.array([1.0, 1.0])
                print(f"✅ Using legacy tanh normalization ([-1, 1])")
            
            print(f"✅ Loaded data info: {self.x_count}x{self.y_count} grid")
            if self.model_metadata:
                print(f"✅ Model metadata: {self.model_metadata.get('latent_dim', '?')}D latent, {self.model_metadata.get('num_shapes', '?')} shapes")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def generate_sdf_at_point(self, latent_x, latent_y):
        """Generate SDF at given latent coordinates"""
        if self.decoder is None:
            return None
            
        # Create latent code
        latent_code = np.array([[latent_x, latent_y]], dtype=np.float32)
        
        # Apply appropriate normalization based on model type
        if self.normalization_type == 'natural_range':
            # Use raw latent codes directly (no normalization)
            # Optionally clamp to exploration range for safety
            latent_code_norm = np.clip(latent_code, self.latent_min, self.latent_max)
        else:
            # Legacy: apply tanh normalization for old models
            latent_code_norm = np.tanh(latent_code)
        
        # Broadcast to match coordinates
        num_coords = self.coords_flat.shape[0]
        batch_latent_codes = np.tile(latent_code_norm, [num_coords, 1])
        
        # Concatenate and predict
        decoder_input = np.concatenate([batch_latent_codes, self.coords_flat], axis=1)
        sdf_values = self.decoder(decoder_input).numpy().flatten()
        
        # Reshape to 2D grid using data.json dimensions
        sdf = sdf_values.reshape(self.y_count, self.x_count)
        
        return sdf
    
    def _load_model_with_compatibility(self, model_path):
        """Load model with TensorFlow version compatibility handling"""
        try:
            # First attempt: standard loading
            return keras.models.load_model(model_path)
        except Exception as e:
            print(f"⚠️  Standard model loading failed: {e}")
            
            # Check if it's the batch_shape issue
            if 'batch_shape' in str(e) or 'InputLayer' in str(e):
                print("🔧 Attempting to fix TensorFlow version compatibility...")
                
                try:
                    # Custom objects to handle deprecated parameters
                    def custom_input_layer(*args, **kwargs):
                        # Replace batch_shape with input_shape if present
                        if 'batch_shape' in kwargs:
                            batch_shape = kwargs.pop('batch_shape')
                            if batch_shape and len(batch_shape) > 1:
                                kwargs['input_shape'] = batch_shape[1:]  # Remove batch dimension
                        
                        # Handle other deprecated parameters
                        deprecated_params = ['dtype', 'sparse', 'ragged']
                        for param in deprecated_params:
                            if param in kwargs:
                                kwargs.pop(param)
                        
                        return tf.keras.layers.InputLayer(*args, **kwargs)
                    
                    # Load with custom objects
                    custom_objects = {
                        'InputLayer': custom_input_layer
                    }
                    
                    return keras.models.load_model(model_path, custom_objects=custom_objects)
                    
                except Exception as e2:
                    print(f"⚠️  Custom loading also failed: {e2}")
                    
                    # Final attempt: load weights manually
                    try:
                        print("🔧 Attempting manual model reconstruction...")
                        return self._reconstruct_model_manually(model_path)
                    except Exception as e3:
                        print(f"❌ Manual reconstruction failed: {e3}")
                        raise e  # Re-raise original error
            else:
                # Re-raise original error if it's not the known compatibility issue
                raise e
    
    def _reconstruct_model_manually(self, model_path):
        """Manually reconstruct model when automatic loading fails"""
        print("🔧 Attempting to reconstruct model from architecture...")
        
        # Try to create a simple decoder model that should work with most TensorFlow versions
        # This is a fallback that creates a basic neural network structure
        
        # Load just the weights if possible
        try:
            # Create a simple model structure that matches typical decoder architecture
            # Input: latent_code (2D) + coordinates (2D) = 4D input
            inputs = tf.keras.layers.Input(shape=(4,), name='decoder_input')
            
            # Simple dense network (adjust layers as needed for your models)
            x = tf.keras.layers.Dense(256, activation='relu')(inputs)
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            outputs = tf.keras.layers.Dense(1, activation='linear')(x)  # SDF output
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs, name='decoder')
            
            # Try to load weights
            try:
                model.load_weights(model_path)
                print("✅ Successfully reconstructed model with loaded weights")
                return model
            except:
                print("⚠️  Could not load weights, using random initialization")
                print("   Model may not produce correct results!")
                return model
                
        except Exception as e:
            print(f"❌ Model reconstruction failed: {e}")
            raise RuntimeError(
                f"Could not load model from {model_path}. "
                "This may be due to TensorFlow version incompatibility. "
                "Try re-exporting the model with the current TensorFlow version."
            )

class PathSelectApp:
    """Main application class for interactive latent space navigation"""
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("PathSelect - Latent SDF Navigator")
        
        # Fonts
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 16)
        self.large_font = pygame.font.Font(None, 32)
        
        # State
        self.model_loaded = False
        self.last_error = None  # Store last error for display
        self.path_points = []
        self.num_frames = 10
        self.generated_sdfs = []
        self.current_sdf = None
        self.show_help = False
        self.show_debug_grid = False  # New: toggle debug grid window
        
        # Background grid for latent space visualization
        self.background_grid = None
        self.background_surface = None
        self.debug_grid_surface = None  # New: separate debug surface
        
        # Output folder - create relative to script location, not working directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_folder = os.path.join(script_dir, "output")
        os.makedirs(self.output_folder, exist_ok=True)
    
    def load_model(self):
        """Load the model and generate background grid"""
        try:
            self.last_error = None  # Clear previous error
            self.model_loaded = self.model_loader.load_model()
            if self.model_loaded:
                self.generate_background_grid()
                print("✅ Model loaded successfully!")
            return self.model_loaded
        except Exception as e:
            self.last_error = str(e)
            self.model_loaded = False
            print(f"❌ Error in load_model: {e}")
            return False
    
    def generate_background_grid(self):
        """Generate SDF grid for background visualization"""
        if not self.model_loaded:
            return
        
        print("Generating background SDF grid...")
        grid_size =5
        self.background_grid = []
        self.debug_grid_sdfs = []  # Store actual SDF images for debug window
        
        # Generate grid points within the latent space
        x_points = np.linspace(self.model_loader.latent_min[0], self.model_loader.latent_max[0], grid_size)
        y_points = np.linspace(self.model_loader.latent_min[1], self.model_loader.latent_max[1], grid_size)
        
        print(f"Generating {grid_size}x{grid_size} SDF grid...")
        for i, lat_y in enumerate(y_points):
            row = []
            debug_row = []
            for j, lat_x in enumerate(x_points):
                print(f"Generating SDF at ({lat_x:.3f}, {lat_y:.3f})...")
                sdf = self.model_loader.generate_sdf_at_point(lat_x, lat_y)
                if sdf is not None:
                    # Store original SDF for debug display
                    debug_row.append(sdf.copy())
                    
                    # Store the actual SDF for background rendering (not just a gray value)
                    row.append(sdf.copy())
                    print(f"  -> SDF range: [{np.min(sdf):.3f}, {np.max(sdf):.3f}], mean: {np.mean(sdf):.3f}, std: {np.std(sdf):.3f}")
                else:
                    row.append(None)  # No SDF available
                    debug_row.append(None)
                    print(f"  -> No SDF generated")
            self.background_grid.append(row)
            self.debug_grid_sdfs.append(debug_row)
        
        # Create background surface and debug surface
        self.create_background_surface(grid_size)
        self.create_debug_grid_surface(grid_size)
        print("Background grid generated successfully")
    
    def create_background_surface(self, grid_size):
        """Create pygame surface from background grid showing actual SDF images"""
        if not self.background_grid:
            return
        
        print("Creating background surface with actual SDF images...")
        
        # Calculate cell size for the grid
        cell_size = 100  # Size of each SDF image in the grid
        grid_surface = pygame.Surface((grid_size * cell_size, grid_size * cell_size))
        grid_surface.fill(WHITE)  # White background
        
        # Render each SDF as an image in the grid
        for i in range(grid_size):
            for j in range(grid_size):
                sdf = self.background_grid[i][j]
                if sdf is not None:
                    # Convert SDF to surface using the same normalization as the main SDF display
                    sdf_clamped = np.clip(sdf, -0.2, 0.2)
                    sdf_norm = (sdf_clamped + 0.2) / 0.4  # Normalize to [0, 1]
                    
                    # Create surface for this SDF
                    sdf_surface = pygame.Surface((self.model_loader.x_count, self.model_loader.y_count))
                    for y in range(self.model_loader.y_count):
                        for x in range(self.model_loader.x_count):
                            val = int(sdf_norm[y, x] * 255)
                            color = (val, val, val)
                            sdf_surface.set_at((x, y), color)
                    
                    # Scale to cell size and blit to grid surface
                    scaled_sdf = pygame.transform.scale(sdf_surface, (cell_size-4, cell_size-4))
                    grid_surface.blit(scaled_sdf, (j * cell_size + 2, i * cell_size + 2))
                    
                    # Add border around each cell
                    rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                    pygame.draw.rect(grid_surface, BLACK, rect, 2)
                    
                    print(f"Grid cell ({i},{j}): rendered SDF image")
                else:
                    # Fill with gray if no SDF
                    rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                    grid_surface.fill(LIGHT_GRAY, rect)
                    pygame.draw.rect(grid_surface, BLACK, rect, 2)
                    print(f"Grid cell ({i},{j}): no SDF, filled with gray")
        
        # Scale to latent view size with smooth scaling
        self.background_surface = pygame.transform.smoothscale(grid_surface, (LATENT_VIEW_WIDTH, LATENT_VIEW_HEIGHT))
        print(f"Background surface created with SDF images: {LATENT_VIEW_WIDTH}x{LATENT_VIEW_HEIGHT}")
    
    def create_debug_grid_surface(self, grid_size):
        """Create detailed debug surface showing actual SDF images"""
        if not self.debug_grid_sdfs:
            return
        
        cell_size = 120  # Size of each SDF image in debug view
        debug_surface = pygame.Surface((grid_size * cell_size, grid_size * cell_size))
        debug_surface.fill(WHITE)
        
        for i in range(grid_size):
            for j in range(grid_size):
                sdf = self.debug_grid_sdfs[i][j]
                if sdf is not None:
                    # Convert SDF to surface
                    sdf_clamped = np.clip(sdf, -0.2, 0.2)
                    sdf_norm = (sdf_clamped + 0.2) / 0.4  # Normalize to [0, 1]
                    
                    # Create mini surface for this SDF
                    mini_surface = pygame.Surface((self.model_loader.x_count, self.model_loader.y_count))
                    for y in range(self.model_loader.y_count):
                        for x in range(self.model_loader.x_count):
                            val = int(sdf_norm[y, x] * 255)
                            color = (val, val, val)
                            mini_surface.set_at((x, y), color)
                    
                    # Scale to cell size and blit to debug surface
                    scaled_mini = pygame.transform.scale(mini_surface, (cell_size-4, cell_size-4))
                    debug_surface.blit(scaled_mini, (j * cell_size + 2, i * cell_size + 2))
                    
                    # Add border
                    rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                    pygame.draw.rect(debug_surface, BLACK, rect, 2)
        
        self.debug_grid_surface = debug_surface
        print("Debug grid surface created")
    
    def screen_to_latent(self, screen_x, screen_y):
        """Convert screen coordinates to latent space coordinates"""
        rel_x = screen_x - 50
        rel_y = screen_y - 50
        
        if 0 <= rel_x <= LATENT_VIEW_WIDTH and 0 <= rel_y <= LATENT_VIEW_HEIGHT:
            # Get the actual latent range from the model
            if hasattr(self.model_loader, 'latent_min') and hasattr(self.model_loader, 'latent_max'):
                latent_range_x = self.model_loader.latent_max[0] - self.model_loader.latent_min[0]
                latent_range_y = self.model_loader.latent_max[1] - self.model_loader.latent_min[1]
                
                latent_x = self.model_loader.latent_min[0] + (rel_x / LATENT_VIEW_WIDTH) * latent_range_x
                latent_y = self.model_loader.latent_min[1] + (rel_y / LATENT_VIEW_HEIGHT) * latent_range_y
            else:
                # Fallback to [-1, 1] for backward compatibility
                latent_x = -1 + (rel_x / LATENT_VIEW_WIDTH) * 2
                latent_y = -1 + (rel_y / LATENT_VIEW_HEIGHT) * 2
            return latent_x, latent_y
        return None
    
    def latent_to_screen(self, latent_x, latent_y):
        """Convert latent coordinates to screen coordinates"""
        # Get the actual latent range from the model
        if hasattr(self.model_loader, 'latent_min') and hasattr(self.model_loader, 'latent_max'):
            latent_range_x = self.model_loader.latent_max[0] - self.model_loader.latent_min[0]
            latent_range_y = self.model_loader.latent_max[1] - self.model_loader.latent_min[1]
            
            norm_x = (latent_x - self.model_loader.latent_min[0]) / latent_range_x
            norm_y = (latent_y - self.model_loader.latent_min[1]) / latent_range_y
            
            screen_x = 50 + norm_x * LATENT_VIEW_WIDTH
            screen_y = 50 + norm_y * LATENT_VIEW_HEIGHT
        else:
            # Fallback to [-1, 1] mapping for backward compatibility
            screen_x = 50 + ((latent_x + 1) / 2) * LATENT_VIEW_WIDTH
            screen_y = 50 + ((latent_y + 1) / 2) * LATENT_VIEW_HEIGHT
        return int(screen_x), int(screen_y)
    
    def interpolate_path(self, num_frames):
        """Generate interpolated points along the entire path with global subdivision"""
        if len(self.path_points) < 2:
            return []
        
        # For a path with multiple segments, distribute frames evenly across the entire path
        total_segments = len(self.path_points) - 1
        
        if total_segments == 1:
            # Simple case: single segment
            start = self.path_points[0]
            end = self.path_points[1]
            
            interpolated = []
            for i in range(num_frames):
                t = i / (num_frames - 1) if num_frames > 1 else 0
                x = start[0] * (1 - t) + end[0] * t
                y = start[1] * (1 - t) + end[1] * t
                interpolated.append((x, y))
            
            return interpolated
        
        else:
            # Multiple segments: distribute frames globally across entire path
            interpolated = []
            
            for i in range(num_frames):
                # Global parameter t from 0 to 1 across entire path
                t_global = i / (num_frames - 1) if num_frames > 1 else 0
                
                # Scale to segment space
                t_scaled = t_global * total_segments
                segment_idx = int(t_scaled)
                
                # Clamp to valid segment range
                if segment_idx >= total_segments:
                    segment_idx = total_segments - 1
                    t_local = 1.0
                else:
                    t_local = t_scaled - segment_idx
                
                # Interpolate within the segment
                start = self.path_points[segment_idx]
                end = self.path_points[segment_idx + 1]
                
                x = start[0] * (1 - t_local) + end[0] * t_local
                y = start[1] * (1 - t_local) + end[1] * t_local
                interpolated.append((x, y))
            
            return interpolated
    
    def generate_path_sdfs(self):
        """Generate SDFs along the interpolated path"""
        if not self.model_loaded or len(self.path_points) < 2:
            return
        
        print(f"Generating {self.num_frames} frames along path...")
        interpolated_points = self.interpolate_path(self.num_frames)
        
        self.generated_sdfs = []
        for i, (lat_x, lat_y) in enumerate(interpolated_points):
            sdf = self.model_loader.generate_sdf_at_point(lat_x, lat_y)
            if sdf is not None:
                self.generated_sdfs.append({
                    'sdf': sdf,
                    'latent': (lat_x, lat_y),
                    'index': i
                })
        
        print(f"Generated {len(self.generated_sdfs)} SDFs")
    
    def save_generated_sdfs(self):
        """Save generated SDFs to JSON format"""
        if not self.generated_sdfs:
            print("No SDFs to save!")
            return
        
        # Start with all metadata from the original data.json
        output_data = {}
        
        # Copy all fields from original data except scalar field data arrays
        for key, value in self.model_loader.data_info.items():
            # Keep metadata fields but exclude actual scalar field data arrays
            if key.startswith('scalar_field_') and isinstance(value, list) and len(value) > 10:
                # This is likely actual SDF data (large array), skip it
                continue
            else:
                # This is metadata (grid dimensions, sizes, model info, etc.), keep it
                output_data[key] = value
        
        # Add generated scalar fields
        for i, sdf_data in enumerate(self.generated_sdfs):
            field_name = f"out_scalar_field_{i:02d}"
            sdf_flat = sdf_data['sdf'].flatten().tolist()
            output_data[field_name] = sdf_flat
        
        # Save to file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(self.output_folder, f"sdf_path_{timestamp}.json")
        
        with open(output_filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Saved {len(self.generated_sdfs)} SDFs to {output_filename}")
        print(f"Preserved {len([k for k in output_data.keys() if not k.startswith('out_scalar_field_')])} metadata fields")
    
    def draw_sdf(self, sdf, x_offset, y_offset, width, height):
        """Draw SDF as a grayscale image with consistent normalization"""
        if sdf is None:
            return
        
        # Apply same normalization as notebook: clamp to [-0.2, 0.2] then grayscale
        sdf_clamped = np.clip(sdf, -0.2, 0.2)
        sdf_norm = (sdf_clamped + 0.2) / 0.4  # Normalize to [0, 1]
        
        # Create surface using data.json dimensions
        sdf_surface = pygame.Surface((self.model_loader.x_count, self.model_loader.y_count))
        
        for i in range(self.model_loader.y_count):
            for j in range(self.model_loader.x_count):
                val = int(sdf_norm[i, j] * 255)
                color = (val, val, val)
                sdf_surface.set_at((j, i), color)
        
        # Scale to desired size
        scaled_surface = pygame.transform.scale(sdf_surface, (width, height))
        self.screen.blit(scaled_surface, (x_offset, y_offset))
    
    def draw_loading_screen(self):
        """Draw loading/error screen"""
        self.screen.fill(WHITE)
        
        # Title
        title = self.large_font.render("PathSelect - Latent SDF Navigator", True, BLACK)
        title_rect = title.get_rect(center=(WINDOW_WIDTH//2, 100))
        self.screen.blit(title, title_rect)
        
        if not self.model_loaded:
            # Error message
            error_msg = self.font.render("Model Loading Failed!", True, RED)
            error_rect = error_msg.get_rect(center=(WINDOW_WIDTH//2, 200))
            self.screen.blit(error_msg, error_rect)
            
            # Instructions
            instructions = [
                "Required folder structure:",
                "  models/decoder_model.h5",
                "  models/latent_codes.npy", 
                "  models/coords_flat.npy",
                "  models/data.json",
                "",
                "Common issues:",
                "• TensorFlow version mismatch (batch_shape error)",
                "• Missing model files",
                "• Corrupted model files",
                "",
                "Solutions:",
                "1. Check console for detailed error messages",
                "2. Try re-exporting model with current TensorFlow version",
                "3. Ensure all model files are present and valid",
                "4. Press R to reload model files",
                "",
                "Current environment:",
                f"  TensorFlow: {tf.__version__}",
                f"  Python: {sys.version.split()[0]}",
                "",
                "Controls:",
                "R - Reload model files",
                "ESC - Exit"
            ]
            
            for i, instruction in enumerate(instructions):
                color = RED if "Required" in instruction or "Common" in instruction or "Solutions" in instruction else BLACK
                if instruction.startswith("  TensorFlow:") or instruction.startswith("  Python:"):
                    color = BLUE
                text = self.small_font.render(instruction, True, color)
                text_rect = text.get_rect(center=(WINDOW_WIDTH//2, 250 + i * 18))
                self.screen.blit(text, text_rect)
            
            # Show detailed error if available
            if hasattr(self, 'last_error') and self.last_error:
                error_y = 250 + len(instructions) * 18 + 20
                error_title = self.small_font.render("Last Error:", True, RED)
                error_title_rect = error_title.get_rect(center=(WINDOW_WIDTH//2, error_y))
                self.screen.blit(error_title, error_title_rect)
                
                # Truncate long error messages
                error_text = self.last_error
                if len(error_text) > 80:
                    error_text = error_text[:77] + "..."
                
                error_msg = self.small_font.render(error_text, True, RED)
                error_msg_rect = error_msg.get_rect(center=(WINDOW_WIDTH//2, error_y + 15))
                self.screen.blit(error_msg, error_msg_rect)
    
    def draw_help_screen(self):
        """Draw help overlay"""
        # Semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(200)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Help content
        help_title = self.large_font.render("HELP - PathSelect Controls", True, WHITE)
        title_rect = help_title.get_rect(center=(WINDOW_WIDTH//2, 100))
        self.screen.blit(help_title, title_rect)
        
        help_content = [
            "MOUSE CONTROLS:",
            "  Left Click - Add path point in latent space",
            "  Right Click - Remove last path point",
            "  Mouse Hover - Preview SDF at cursor position",
            "",
            "KEYBOARD CONTROLS:",
            "  SPACE - Generate SDF sequence along path",
            "  S - Save generated SDFs to output folder",
            "  C - Clear all path points",
            "  UP/DOWN - Increase/decrease number of frames",
            "  R - Reload model files",
            "  H - Toggle this help screen",
            "  ESC - Exit application",
            "",
            "LATENT SPACE:",
            "  Green circles - Original training shapes",
            "  Blue circles - Your path points",
            "  Red lines - Path connections",
            "  Grid shows learned latent coordinate space",
            "",
            "FRAMES:",
            "  Number of frames = exact number of SDFs generated",
            "  Frames are distributed evenly across entire path",
            "",
            "Press H again to close this help"
        ]
        
        for i, line in enumerate(help_content):
            color = YELLOW if line.endswith(":") else WHITE
            text = self.small_font.render(line, True, color)
            text_rect = text.get_rect(center=(WINDOW_WIDTH//2, 160 + i * 18))
            self.screen.blit(text, text_rect)
    
    def draw_main_ui(self):
        """Draw the main user interface"""
        self.screen.fill(WHITE)
        
        # Draw latent space view with background
        if self.background_surface:
            self.screen.blit(self.background_surface, (50, 50))
        else:
            pygame.draw.rect(self.screen, LIGHT_GRAY, (50, 50, LATENT_VIEW_WIDTH, LATENT_VIEW_HEIGHT))
            # Add text to show background not loaded
            no_bg_text = self.small_font.render("Background not loaded", True, RED)
            self.screen.blit(no_bg_text, (60, 60))
        
        pygame.draw.rect(self.screen, BLACK, (50, 50, LATENT_VIEW_WIDTH, LATENT_VIEW_HEIGHT), 2)
        
        # Draw subtle grid lines
        for i in range(1, 5):
            x_pos = 50 + (i / 4) * LATENT_VIEW_WIDTH
            pygame.draw.line(self.screen, (180, 180, 180), (x_pos, 50), (x_pos, 50 + LATENT_VIEW_HEIGHT))
            y_pos = 50 + (i / 4) * LATENT_VIEW_HEIGHT
            pygame.draw.line(self.screen, (180, 180, 180), (50, y_pos), (50 + LATENT_VIEW_WIDTH, y_pos))
        
        # Draw center lines (slightly more visible)
        center_x = 50 + LATENT_VIEW_WIDTH // 2
        center_y = 50 + LATENT_VIEW_HEIGHT // 2
        pygame.draw.line(self.screen, (150, 150, 150), (center_x, 50), (center_x, 50 + LATENT_VIEW_HEIGHT), 1)
        pygame.draw.line(self.screen, (150, 150, 150), (50, center_y), (50 + LATENT_VIEW_WIDTH, center_y), 1)
        
        # Draw learned latent codes
        if self.model_loader.normalization_type == 'natural_range':
            learned_codes = self.model_loader.latent_codes
        else:
            learned_codes = np.tanh(self.model_loader.latent_codes)
        
        for i, code in enumerate(learned_codes):
            screen_x, screen_y = self.latent_to_screen(code[0], code[1])
            pygame.draw.circle(self.screen, GREEN, (screen_x, screen_y), 8)
            # Add white outline for better visibility on background
            pygame.draw.circle(self.screen, WHITE, (screen_x, screen_y), 9, 2)
            label = self.small_font.render(f"S{i}", True, BLACK)
            label_bg = pygame.Surface((label.get_width() + 4, label.get_height() + 2))
            label_bg.fill(WHITE)
            label_bg.set_alpha(200)
            self.screen.blit(label_bg, (screen_x + 8, screen_y - 10))
            self.screen.blit(label, (screen_x + 10, screen_y - 8))
        
        # Draw path points
        for i, (lat_x, lat_y) in enumerate(self.path_points):
            screen_x, screen_y = self.latent_to_screen(lat_x, lat_y)
            pygame.draw.circle(self.screen, BLUE, (screen_x, screen_y), 6)
            # Add white outline for better visibility
            pygame.draw.circle(self.screen, WHITE, (screen_x, screen_y), 7, 2)
            label = self.small_font.render(f"P{i}", True, BLACK)
            label_bg = pygame.Surface((label.get_width() + 4, label.get_height() + 2))
            label_bg.fill(WHITE)
            label_bg.set_alpha(200)
            self.screen.blit(label_bg, (screen_x + 6, screen_y - 10))
            self.screen.blit(label, (screen_x + 8, screen_y - 8))
        
        # Draw path lines with better visibility
        if len(self.path_points) > 1:
            screen_points = [self.latent_to_screen(lat_x, lat_y) for lat_x, lat_y in self.path_points]
            # Draw white outline for path
            for i in range(len(screen_points) - 1):
                pygame.draw.line(self.screen, WHITE, screen_points[i], screen_points[i + 1], 4)
            # Draw red path on top
            pygame.draw.lines(self.screen, RED, False, screen_points, 2)
        
        # Draw current SDF preview
        if self.current_sdf is not None:
            self.draw_sdf(self.current_sdf, 700, 50, SDF_VIEW_WIDTH, SDF_VIEW_HEIGHT)
            pygame.draw.rect(self.screen, BLACK, (700, 50, SDF_VIEW_WIDTH, SDF_VIEW_HEIGHT), 2)
        
        # Draw debug grid window if enabled
        if self.show_debug_grid and self.debug_grid_surface:
            # Draw semi-transparent background
            debug_bg = pygame.Surface((620, 620))
            debug_bg.set_alpha(240)
            debug_bg.fill(WHITE)
            self.screen.blit(debug_bg, (10, 10))
            
            # Draw debug grid (scaled down to fit)
            debug_scaled = pygame.transform.scale(self.debug_grid_surface, (600, 600))
            self.screen.blit(debug_scaled, (20, 20))
            
            # Add border and title
            pygame.draw.rect(self.screen, BLACK, (10, 10, 620, 620), 3)
            title = self.font.render("Debug: 5x5 SDF Grid (Press D to toggle)", True, BLACK)
            self.screen.blit(title, (15, 635))
        
        # Draw UI text
        title = self.font.render("PathSelect - Latent SDF Navigator", True, BLACK)
        self.screen.blit(title, (50, 10))
        
        # Model info using data.json
        if hasattr(self.model_loader, 'x_count'):
            info_parts = [f"Grid: {self.model_loader.x_count}x{self.model_loader.y_count}"]
            
            # Add normalization type info
            if hasattr(self.model_loader, 'normalization_type'):
                if self.model_loader.normalization_type == 'natural_range':
                    range_info = f"Range: [{self.model_loader.latent_min[0]:.2f},{self.model_loader.latent_min[1]:.2f}] to [{self.model_loader.latent_max[0]:.2f},{self.model_loader.latent_max[1]:.2f}]"
                    info_parts.append(range_info)
                else:
                    info_parts.append("Range: [-1,1] (tanh)")
            
            if self.model_loader.model_metadata:
                meta = self.model_loader.model_metadata
                if 'num_shapes' in meta:
                    info_parts.append(f"Shapes: {meta['num_shapes']}")
                if 'latent_dim' in meta:
                    info_parts.append(f"Latent: {meta['latent_dim']}D")
            
            # Split info across multiple lines if too long
            info_text1 = " | ".join(info_parts[:2])
            info_text2 = " | ".join(info_parts[2:]) if len(info_parts) > 2 else ""
            
            text1 = self.small_font.render(info_text1, True, BLACK)
            self.screen.blit(text1, (50, 670))
            if info_text2:
                text2 = self.small_font.render(info_text2, True, BLACK)
                self.screen.blit(text2, (50, 685))
        
        # Controls info
        frames_text = self.font.render(f"Number of Frames: {self.num_frames}", True, BLACK)
        self.screen.blit(frames_text, (700, 470))
        
        # Debug info for background grid
        if hasattr(self, 'background_grid') and self.background_grid:
            bg_info = f"Background: {len(self.background_grid)}x{len(self.background_grid[0])} grid"
            bg_text = self.small_font.render(bg_info, True, BLUE)
            self.screen.blit(bg_text, (700, 490))
            
            # Show some grid values for debugging
            if len(self.background_grid) > 0:
                sample_values = [str(self.background_grid[i][0]) for i in range(min(3, len(self.background_grid)))]
                values_text = f"Values: {', '.join(sample_values)}"
                values_render = self.small_font.render(values_text, True, BLUE)
                self.screen.blit(values_render, (700, 505))
        else:
            no_bg_text = self.small_font.render("No background grid generated", True, RED)
            self.screen.blit(no_bg_text, (700, 490))
        
        if self.generated_sdfs:
            sdf_count_text = self.font.render(f"Generated SDFs: {len(self.generated_sdfs)}", True, GREEN)
            self.screen.blit(sdf_count_text, (700, 450))
        
        # Quick controls
        controls = [
            "H - Help",
            "D - Debug Grid",
            "SPACE - Generate",
            "S - Save",
            "C - Clear",
            "ESC - Exit"
        ]
        
        for i, control in enumerate(controls):
            text = self.small_font.render(control, True, BLACK)
            self.screen.blit(text, (700, 530 + i * 18))
    
    def handle_mouse_hover(self, mouse_pos):
        """Handle mouse hover to show preview"""
        if not self.model_loaded:
            return
            
        latent_coords = self.screen_to_latent(mouse_pos[0], mouse_pos[1])
        if latent_coords:
            lat_x, lat_y = latent_coords
            self.current_sdf = self.model_loader.generate_sdf_at_point(lat_x, lat_y)
    
    def run(self):
        """Main application loop"""
        clock = pygame.time.Clock()
        running = True
        
        # Try to load model on startup
        self.load_model()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_h:
                        self.show_help = not self.show_help
                    elif event.key == pygame.K_d:  # New: toggle debug grid
                        self.show_debug_grid = not self.show_debug_grid
                    elif event.key == pygame.K_r:
                        print("Reloading model...")
                        self.load_model()
                    elif self.model_loaded:
                        if event.key == pygame.K_UP:
                            self.num_frames = min(100, self.num_frames + 1)
                        elif event.key == pygame.K_DOWN:
                            self.num_frames = max(2, self.num_frames - 1)
                        elif event.key == pygame.K_SPACE:
                            self.generate_path_sdfs()
                        elif event.key == pygame.K_s:
                            self.save_generated_sdfs()
                        elif event.key == pygame.K_c:
                            self.path_points.clear()
                            self.generated_sdfs.clear()
                
                elif event.type == pygame.MOUSEBUTTONDOWN and self.model_loaded:
                    if event.button == 1:  # Left click
                        latent_coords = self.screen_to_latent(event.pos[0], event.pos[1])
                        if latent_coords:
                            self.path_points.append(latent_coords)
                    elif event.button == 3:  # Right click
                        if self.path_points:
                            self.path_points.pop()
                
                elif event.type == pygame.MOUSEMOTION and self.model_loaded:
                    self.handle_mouse_hover(event.pos)
            
            # Draw appropriate screen
            if self.show_help:
                self.draw_help_screen()
            elif self.model_loaded:
                self.draw_main_ui()
            else:
                self.draw_loading_screen()
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()

def main():
    """Main entry point"""
    print("PathSelect - Latent SDF Navigator")
    print("=" * 40)

    
    # try:
    pygame.init()
    app = PathSelectApp()
    app.run()
    # except Exception as e:
    #     print(f"Error: {e}")
    #     input("Press Enter to exit...")
    #     sys.exit(1)

if __name__ == "__main__":
    main()
