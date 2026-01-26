import cv2
import numpy as np
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog





################################################################################################################
#............................................... Low-Level Features............................................#
################################################################################################################

# ...............................................  Color features  ........................................... #

#------------------------------------------
# 1. RGB + HSV color histograms (per frame)
#------------------------------------------
'''
def color_hist_features(frame, bins=32):
    # RGB histogram
    rgb_hist = cv2.calcHist([frame], [0,1,2], None,
                            [bins, bins, bins],
                            [0,256, 0,256, 0,256])
    rgb_hist = cv2.normalize(rgb_hist, rgb_hist).flatten()

    # HSV histogram
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv], [0,1,2], None,
                            [bins, bins, bins],
                            [0,180, 0,256, 0,256])
    hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten()

    return np.hstack([rgb_hist, hsv_hist])

'''
def color_hist_features(frame, bins=32):

    # Ensure frame is in uint8 format (0–255) for OpenCV operations
    frame_u8 = (frame * 255).astype(np.uint8)

    # ---------------- RGB histogram ----------------
    rgb_hist = cv2.calcHist([frame_u8], [0,1,2], None,
                            [bins, bins, bins],
                            [0,256, 0,256, 0,256])
    rgb_hist = cv2.normalize(rgb_hist, rgb_hist).flatten()

    # ---------------- HSV histogram ----------------
    hsv = cv2.cvtColor(frame_u8, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv], [0,1,2], None,
                            [bins, bins, bins],
                            [0,180, 0,256, 0,256])
    hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten()

    return np.hstack([rgb_hist, hsv_hist])


#--------------------------------------------
# 2. Average color distribution across video
#-------------------------------------------

def video_color_features(frames):
    hist_list = []
    moment_list = []

    for f in frames:
        hist_list.append(color_hist_features(f))
        moment_list.append(color_moments(f))

    avg_hist = np.mean(hist_list, axis=0)
    avg_mom  = np.mean(moment_list, axis=0)

    return np.hstack([avg_hist, avg_mom])


#--------------------------------------------
# 3. Color moments (mean, variance, skewness)
#--------------------------------------------
 
def color_moments(frame):
    img = frame.reshape(-1, 3)

    mean = np.mean(img, axis=0)
    var  = np.var(img, axis=0)
    sk   = skew(img, axis=0)

    return np.hstack([mean, var, sk])


# ...............................................  Texture features  ........................................... #

#------------------------------------------
# 1. Gray Level Co-occurrence Matrix (GLCM)
#------------------------------------------

def glcm_features(frame):
    gray = gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(gray,
                        distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256,
                        symmetric=True,
                        normed=True)

    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    feats = []

    for p in props:
        feats.extend(graycoprops(glcm, p).flatten())

    return np.array(feats)

#------------------------------------------
# 2. Local Binary Patterns (LBP)
#------------------------------------------

def lbp_features(frame, P=8, R=1):
    gray = gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P, R, method="uniform")

    hist, _ = np.histogram(lbp.ravel(),
                           bins=np.arange(0, P+3),
                           range=(0, P+2),
                           density=True)

    return hist



#------------------------------------------
# 3. Gabor filter responses
#------------------------------------------

def gabor_features(frame):
    gray = gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    feats = []

    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        kernel = cv2.getGaborKernel((21,21), 5.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        feats.append(filtered.mean())
        feats.append(filtered.var())

    return np.array(feats)


#------------------------------------------
# 4. Video-level texture features (average across frames)
#------------------------------------------

def video_texture_features(frames):
    glcm_list, lbp_list, gabor_list = [], [], []

    for f in frames:
        glcm_list.append(glcm_features(f))
        lbp_list.append(lbp_features(f))
        gabor_list.append(gabor_features(f))

    glcm_avg = np.mean(glcm_list, axis=0)
    lbp_avg = np.mean(lbp_list, axis=0)
    gabor_avg = np.mean(gabor_list, axis=0)

    return np.hstack([glcm_avg, lbp_avg, gabor_avg])


# ...............................................  Shape features  ........................................... #

#------------------------------------------
# 1. Edge histograms using Canny edge detection
#------------------------------------------

def edge_features(frame):
    gray = gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    hist, _ = np.histogram(edges.ravel(), bins=2, range=(0,256), density=True)
    return hist   # edge vs non-edge distribution

#------------------------------------------
# 2. Contour-based features
#------------------------------------------
def contour_features(frame):
    gray = gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contours]
    perimeters = [cv2.arcLength(c, True) for c in contours]

    if len(areas) == 0:
        return np.zeros(4)

    return np.array([
        len(contours),
        np.mean(areas),
        np.mean(perimeters),
        np.max(areas)
    ])

#------------------------------------------
# 3. HOG (Histogram of Oriented Gradients)
#------------------------------------------

def hog_features(frame):
    gray = gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    hog_feat = hog(gray,
                   orientations=9,
                   pixels_per_cell=(16,16),
                   cells_per_block=(2,2),
                   block_norm='L2-Hys',
                   visualize=False,
                   feature_vector=True)
    return hog_feat

#------------------------------------------
# 4. Video-level shape features
#------------------------------------------
def video_shape_features(frames):
    edge_list, contour_list, hog_list = [], [], []

    for f in frames:
        edge_list.append(edge_features(f))
        contour_list.append(contour_features(f))
        hog_list.append(hog_features(f))

    edge_avg = np.mean(edge_list, axis=0)
    contour_avg = np.mean(contour_list, axis=0)
    hog_avg = np.mean(hog_list, axis=0)

    return np.hstack([edge_avg, contour_avg, hog_avg])


# ...............................................  Motion features  ........................................... #

#------------------------------------------
# 1. Frame Differencing
# – Absolute difference between consecutive frames
# – Statistical measures of frame differences
# – Motion intensity histograms
#------------------------------------------

#Temporal features

#1. Statistical measures of feature sequences
def temporal_stats(feature_sequence):
    feature_sequence = np.array(feature_sequence)

    mean_feat = np.mean(feature_sequence, axis=0)
    std_feat  = np.std(feature_sequence, axis=0)
    min_feat  = np.min(feature_sequence, axis=0)
    max_feat  = np.max(feature_sequence, axis=0)

    return np.hstack([mean_feat, std_feat, min_feat, max_feat])

#2. Frame-to-frame variation analysis

def frame_variation(feature_sequence):
    diffs = []

    for i in range(1, len(feature_sequence)):
        d = np.abs(feature_sequence[i] - feature_sequence[i-1])
        diffs.append(d)

    diffs = np.array(diffs)

    return np.hstack([
        np.mean(diffs, axis=0),
        np.std(diffs, axis=0)
    ])

#3. Temporal gradients & patterns
def temporal_gradient(feature_sequence):
    feature_sequence = np.array(feature_sequence)
    grad = np.gradient(feature_sequence, axis=0)

    return np.hstack([
        np.mean(grad, axis=0),
        np.std(grad, axis=0)
    ])

# 4. Example: motion-based temporal features

def temporal_motion_features(frames):
    motion_vals = []

    for i in range(1, len(frames)):
        g1 = cv2.cvtColor((frames[i-1] * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor((frames[i]   * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(g1, g2)
        motion_vals.append([np.mean(diff)])

    motion_vals = np.array(motion_vals)

    stats = temporal_stats(motion_vals)
    var   = frame_variation(motion_vals)
    grad  = temporal_gradient(motion_vals)

    return np.hstack([stats, var, grad])



