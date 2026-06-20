class NodeEngine:
    def __init__(self, registry):
        """
        registry: dictionary mapping Node Type (string) to a Python class/function.
        Các hàm node cần có signature: execute(self, **kwargs) -> dict
        """
        self.registry = registry

    def resolve_inputs(self, node_inputs, context):
        """
        Phân giải các biến động (bắt đầu bằng @) từ context.
        Ví dụ: "@align_1.image" -> context["align_1"]["image"]
        """
        resolved = {}
        for k, v in node_inputs.items():
            if isinstance(v, str) and v.startswith("@"):
                path = v[1:].split(".")
                val = context
                try:
                    for p in path:
                        val = val[p]
                    resolved[k] = val
                except KeyError:
                    raise ValueError(f"Không tìm thấy biến '{v}' trong Context.")
            else:
                resolved[k] = v
        return resolved

    def execute_pipeline(self, pipeline_config, initial_context):
        """
        pipeline_config: list các node config từ JSON.
        initial_context: dict chứa các input ban đầu (vd: {"input": {"image": img, "answers": ans}})
        """
        context = initial_context.copy()
        
        for node_cfg in pipeline_config:
            node_id = node_cfg.get("id")
            
            # Bỏ qua node nếu đã có kết quả cache hợp lệ (không lỗi) trong context
            if node_id in context and not context[node_id].get("error"):
                continue
                
            node_type = node_cfg.get("type")
            inputs_cfg = node_cfg.get("inputs", {})
            params_cfg = node_cfg.get("params", {})
            
            if node_type not in self.registry:
                raise ValueError(f"Node type '{node_type}' không tồn tại trong Registry.")
                
            node_class_or_func = self.registry[node_type]
            
            # Giải quyết các input động từ context
            resolved_inputs = self.resolve_inputs(inputs_cfg, context)
            
            # Gộp chung static params và dynamic inputs để truyền vào hàm execute
            final_kwargs = {**params_cfg, **resolved_inputs}
            
            # Thực thi Node
            # Support class based nodes with execute() or simple functions
            if hasattr(node_class_or_func, "execute"):
                instance = node_class_or_func()
                outputs = instance.execute(**final_kwargs)
            else:
                outputs = node_class_or_func(**final_kwargs)
                
            if not isinstance(outputs, dict):
                raise TypeError(f"Node '{node_id}' ({node_type}) phải trả về một Dictionary.")
                
            # Lưu kết quả vào context
            context[node_id] = outputs
            
        return context
