# Traffic Sign Detection System

## 1. Software Features

### 1.1 Real-Time Video Processing
- Read local video files and perform object detection and text recognition frame by frame
- Support pause/resume functionality, controllable via on-screen buttons or the spacebar
- Automatically output annotated video files (output.avi) with detection bounding boxes and labels
- One-click screen recording of the entire software interface to save the complete detection demonstration process

### 1.2 Dual-Model Object Detection
- **trafficSign.pt**: Traffic sign detection model covering speed limit signs, prohibition signs, warning signs, and various other categories, with a confidence threshold ≥ 0.70
- **roadSign.pt**: Road guide sign detection model identifying blue-on-white or green-on-white directional road signs, with a confidence threshold ≥ 0.80
- Both models run independently; traffic sign and road sign detection do not interfere with each other
- Support CUDA GPU acceleration and compatible with CPU-only operation mode

### 1.3 OCR Text Recognition
- Based on the EasyOCR deep learning framework, supporting mixed Chinese Simplified and English recognition
- Perform OCR every 5 frames to balance recognition quality and runtime performance
- Text confidence threshold ≥ 0.50 to filter out low-quality recognition results
- Multiple text regions on the same road sign are concatenated with spaces; different road signs are separated by double line breaks
- Recognition results are displayed in the lower-right corner of the interface with a purple highlighted background

### 1.4 PyQt Visualization Interface
- **Left Video Area**: Real-time video stream display with overlaid detection results — traffic signs in **blue boxes** and road signs in **green boxes**; the title bar integrates pause and recording buttons
- **Top-Right Traffic Sign Area**: Cropped traffic sign images arranged from left to right, with an orange title and white-background cards
- **Bottom-Right Road Sign Area**: Cropped road sign images arranged from left to right, with OCR recognition text displayed below each image, featuring a green title and yellow background
- Detected objects continuously appear in the right panels; objects disappear from view after 1 second of absence to avoid flickering
- Traffic signs and road signs each have independent IoU-based cross-frame matching and deduplication; duplicate objects do not reappear
- The display count on the right follows a **monotonic increment policy** — new objects are added upon detection, and old objects disappear automatically after timeout
- During pause mode, all detected objects remain visible without disappearing, facilitating close observation

---

## 2. Technical Workflow

### Overall Processing Pipeline
`Video Input → Frame-by-Frame Reading → YOLO Dual-Model Parallel Inference → Confidence Filtering → Crop Object Regions with Margin Expansion → CLAHE Enhancement & HSV Color Segmentation for Road Signs → Inter-Frame Smoothing Filtering → IoU-Based Cross-Frame Target Matching & Deduplication → EasyOCR Text Recognition → PyQt Multi-Threaded Rendering → Annotated Video Output`

### Road Sign Precise Preservation Workflow
`YOLO-Detected Road Sign Region → CLAHE Adaptive Histogram Equalization (LAB Color Space, L-channel Enhancement) → HSV Color Space Conversion → Blue-Range & Green-Range Color Segmentation → Morphological Closing to Connect Fragments → Morphological Opening to Remove Noise → Find Largest External Contour → Contour Area Filtering (discard if < 5% of image area) → Polygon Approximation to Determine Quadrilateral → If Quadrilateral: Perspective Transform for Rectification → If Not: Minimum Bounding Rectangle Crop → Output Precisely Preserved Road Sign Image`

### Object Tracking and Deduplication Mechanism
`Receive Current Frame Detection Boxes → Compute IoU with Each Object in Existing Object Dictionary → If IoU Exceeds Threshold: Match Success, Update Object Image & Position → If Match Fails: Add as New Object → Traffic Signs Use a Lower IoU Threshold (0.1) to Tolerate Small-Object Jitter → Road Signs Use a Standard IoU Threshold (0.3) → Periodically Clear Objects Not Updated Beyond the Display Timeout (1 second) → During Pause Mode: Suspend Timeout Counting, Keep All Objects Visible`

### Inter-Frame Smoothing Filtering
`Maintain Historical Records of the Last 5 Frames for Road Sign Detections → For Each Candidate Object in the Current Frame, Count Its Appearances in Historical Frames → If Frequency ≥ 3 → Classify as a Stable Object and Send to Display Module → Effectively Filters Out Single-Frame False Positives and Occasional Missed Detections → Traffic Signs Do Not Undergo Inter-Frame Smoothing to Maintain Real-Time Responsiveness`

### Key Technologies
- **PyTorch + YOLO**: Deep learning-based object detection framework using pre-trained models for transfer learning
- **OpenCV**: Handles image reading, color space conversion, morphological operations, contour detection, perspective transformation, and video encoding/writing
- **EasyOCR**: CRNN-based end-to-end text detection and recognition supporting 80+ languages
- **CLAHE**: Contrast-limited adaptive histogram equalization applied to the L-channel in LAB color space to enhance road sign contrast while suppressing noise amplification
- **IoU Algorithm**: Computes the ratio of intersection area to union area of two bounding boxes to determine if they correspond to the same object
- **HSV Color Space**: Converts images from BGR to HSV and extracts specific color regions using hue and saturation thresholds
- **Morphological Operations**: Closing (dilation followed by erosion) to connect adjacent fragments, followed by opening (erosion followed by dilation) to remove isolated noise
- **PyQt5**: Uses QMainWindow as the main window, QSplitter for resizable left-right split layout, QTimer for periodic UI updates, and Queue for thread-safe data transfer
- **Sliding Window Voting**: Maintains a fixed-length detection history queue and determines target stability via a voting mechanism

---

## 3. File Structure and Descriptions

| File/Directory | Description |
|----------------|-------------|
| `main.py` | Program entry point and main loop: video reading, detector initialization, processing pipeline scheduling, and PyQt application startup |
| `display_window.py` | PyQt interface class: encapsulates video display widgets, detection result grid layouts, OCR text display areas, pause/recording button interactions, and object tracking/deduplication algorithms |
| `image_utils.py` | Image processing toolkit: traffic sign region extraction, road sign region extraction, road sign color preservation, CLAHE normalization enhancement, and inter-frame smoothing filtering functions |
| `detector.py` | YOLO detector wrapper class: model loading, single-frame inference, and detection result return |
| `visualizer.py` | Visualization utility class: bounding box drawing, Chinese label rendering, and multi-model color schemes |
| `ocr_detector.py` | OCR detector wrapper class: EasyOCR-based text detection and recognition, result formatting |
| `nms_utils.py` | Non-maximum suppression utility: deduplication and merging of OCR recognition results |
| `trafficSign.pt` | Traffic sign YOLO model weight file |
| `roadSign.pt` | Road sign YOLO model weight file |

---

## 4. Potential Application Scenarios

- **Intelligent Transportation Systems**: Deployed on in-vehicle terminals to recognize road signs and guide signs in real time, providing navigation assistance and safety alerts to drivers
- **Autonomous Driving Perception**: Serves as a component of the visual perception module for autonomous vehicles, supplying traffic sign categories, locations, and textual information from guide signs
- **Road Facility Inspection**: Mounted on inspection vehicles to automatically detect road sign integrity, cleanliness, and occlusions, generating maintenance reports
- **High-Definition Map Updating**: Processes road video data in bulk to extract spatial locations and semantic information of signs for automated map data updates
- **Driver Training Analysis**: Reviews learner driver videos to analyze their observation and reaction to traffic signs, assisting instructional evaluation
- **Smart City Management**: Establishes digital archives of traffic signs for automated inventory management and status monitoring of urban road assets
- **Data Annotation Assistance**: Automatically generates bounding box annotation data for traffic signs and road signs, reducing manual labeling costs and accelerating model training iteration
