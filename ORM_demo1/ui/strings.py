# Contains all static text strings used in the application interface.

APP_NAME = "Lily Image Editor"

# Explorer
LBL_EXPLORER = "EXPLORER"
LBL_IMPORTED_FILES = "Imported Files"
MENU_REMOVE_FOLDER = "Remove Folder from Explorer"

# File Menu & Toolbar
MENU_FILE = "File"
MENU_THEME = "Theme"
THEME_LIGHT = "Light"
THEME_DARK = "Dark"
THEME_WARM = "Warm"
THEME_COLD = "Cold"
MENU_IMPORT_FOLDER = "Import Folder"
MENU_IMPORT_FILE = "Import File"
MENU_SAVE = "Save"
MENU_SAVE_AS = "Save As..."
MENU_SAVE_ALL = "Save All"
MENU_EXIT = "Exit"

# Center Panel / Image View
LBL_NO_IMAGE = "No image loaded\nOpen an image to start"
LBL_UNTITLED = "Untitled-1"

# Dialogs & Messages
MSG_SAVE_CONFIRM_TITLE = "Confirm Save"
MSG_SAVE_CONFIRM_TEXT = "Are you sure you want to save? (This will overwrite the existing file)"
MSG_SAVE_SUCCESS = "Image saved successfully!"
MSG_SAVE_AS_TITLE = "Save Image As"
MSG_SAVE_AS_SUCCESS = "Image saved to:\n{}"
MSG_SAVE_ERROR_TITLE = "Save Error"
MSG_NO_IMAGE_WARNING = "No image selected to save."
MSG_UNSAVED_TAB = "File '{}' has unsaved changes.\nDo you want to save them before closing?"
MSG_UNSAVED_APP_EXIT = "You have {} unsaved files.\nDo you want to save them before exiting?"
MSG_HISTORY_LIMIT = "Edit history limit reached (maximum {} steps after the original image)."
MSG_CACHE_WRITE_FAILED = "Unable to save image history cache (disk full or write permission error)."
MSG_RESET_CONFIRM = (
    "Reset image to original (initial step)?\n"
    "Subsequent edit steps will be removed from history. This action cannot be undone."
)

# History (right panel top row)
BTN_UNDO = "<"
BTN_REDO = ">"
BTN_RESET_ALL = "Reset All"

# Basic Buttons
BTN_RED = "R"
BTN_GREEN = "G"
BTN_BLUE = "B"
BTN_TO_GRAY = "Grayscale"
BTN_CROP_CENTER = "Center Crop (1/4)"
BTN_ANIMATION = "Run Animation"
BTN_STOP_ANIMATION = "Stop Animation"

# Transform Buttons
BTN_NEGATIVE = "Invert (Negative)"
BTN_LOG = "Log Transform"
BTN_GAMMA = "Power-Law (Gamma)"
BTN_HIST_EQUAL = "Histogram Equalization"
BTN_LOCAL_ENHANCE = "Local Enhancement"
BTN_CONTRAST_STRETCH = "Contrast Stretching"
BTN_PIECEWISE_LINEAR = "Piecewise Linear"

# Filter Buttons
BTN_CONVOLUTION = "Convolution"
BTN_MEAN = "Mean Filter"
BTN_GAUSSIAN = "Gaussian Filter"
BTN_BOX = "Box Filter"
BTN_LOWPASS = "Low-pass Filter"
BTN_MEDIAN = "Median Filter"
BTN_SOBEL = "Sobel Filter"
BTN_ROBERT = "Robert Filter"
BTN_PREWITT = "Prewitt Filter"
BTN_MIN = "Min Filter"
BTN_MAX = "Max Filter"
BTN_MIDPOINT = "Midpoint Filter"
BTN_COMBINED = "Combined Spatial"

# Sliders
LBL_KERNEL_SIZE = "Kernel Size: {}"
LBL_GAMMA_VAL = "Gamma Value: {}"
LBL_C_VAL = "Constant C: {}"
LBL_LOW_VAL = "Low Threshold (r1): {}"
LBL_HIGH_VAL = "High Threshold (r2): {}"

LBL_CH1 = "CHAPTER 1"
LBL_CH2 = "CHAPTER 2"

# Chapter 3: Frequency Domain
LBL_FREQ_DOMAIN = "CHAPTER 3"
LBL_D0_VAL = "Cut-off Frequency (D0): {}"
LBL_ORDER_N_VAL = "Butterworth Order (n): {}"

BTN_MAGNITUDE = "Magnitude Spectrum"
BTN_IDEAL_LPF = "Ideal Lowpass"
BTN_BUTTERWORTH_LPF = "Butterworth Lowpass"
BTN_GAUSSIAN_LPF = "Gaussian Lowpass"
BTN_IDEAL_HPF = "Ideal Highpass"
BTN_BUTTERWORTH_HPF = "Butterworth Highpass"
BTN_GAUSSIAN_HPF = "Gaussian Highpass"
BTN_FREQ_LAPLACIAN = "Laplacian (Freq)"

# Chapter 4: Morphology
LBL_MORPHOLOGY = "CHAPTER 4: MORPHOLOGY"
BTN_EROSION = "Erosion"
BTN_DILATION = "Dilation"
BTN_OPENING = "Opening"
BTN_CLOSING = "Closing"

# Chapter 5: Segmentation
LBL_SEGMENTATION = "CHAPTER 5: SEGMENTATION"
