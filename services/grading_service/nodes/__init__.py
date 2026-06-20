from services.grading_service.nodes.cv_nodes import DocumentAlignerNode, AdaptiveThresholdNode, BlockExtractorNode, ImageStandardizerNode, ONNXRestorationNode
from services.grading_service.nodes.logic_nodes import (
    Heuristic120Node,
    SBDReaderNode, SBDVisualizerNode,
    BubbleGridDetectorNode, MCQScorerNode, MCQVisualizerNode
)

NODE_CLASS_MAPPINGS = {
    "ImageStandardizer": ImageStandardizerNode,
    "ONNXRestoration": ONNXRestorationNode,
    "DocumentAligner": DocumentAlignerNode,
    "AdaptiveThreshold": AdaptiveThresholdNode,
    "BlockExtractor": BlockExtractorNode,
    
    "Heuristic120": Heuristic120Node,
    
    "SBDReader": SBDReaderNode,
    "SBDVisualizer": SBDVisualizerNode,
    
    "BubbleGridDetector": BubbleGridDetectorNode,
    "MCQScorer": MCQScorerNode,
    "MCQVisualizer": MCQVisualizerNode
}
