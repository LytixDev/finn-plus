# Wrapper around ONNX models introducing convenience methods to access QONNX and
# FINN specifics
# QONNX arbitrary precision datatype annotations
# Use "greatest common divisor" in folding calculations
import math

# Custom folding is specified and saved as YAML instead of JSON
import yaml
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

# Get QONNX CustomOp instance for the given ONNX node, if it exists.
from qonnx.custom_op.registry import getCustomOp

# Cleanup transformation giving unique names to all nodes reflecting the op-type
# and graph ordering
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes

# QONNX type and shape inference transformations
from qonnx.transformation.infer_shapes import InferShapes

# Configuration for FINN dataflow builds passed through the build steps by the
# FINN frontend
from finn.builder.build_dataflow_config import DataflowBuildConfig

# ONNX operator to FINN HWCustomOp conversion steps, also inferring custom-ops
# from patterns of operators
from finn.transformation.fpgadataflow import convert_to_hw_layers as hardware
from finn.transformation.fpgadataflow.attention import InferScaledDotProductAttention
from finn.transformation.fpgadataflow.replicate_stream import InferReplicateStream

# Reuse FINN auto-folding functionality to build folding of attention operators
from finn.transformation.fpgadataflow.set_folding import (
    AnnotateCycles,
    SetFolding,
    common_divisors,
    dataflow_performance,
)
from finn.transformation.general import ApplyConfig
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten

# FINN Streamlining transformations still required during hardware conversion
from finn.transformation.streamline import RoundAndClipThresholds
from finn.transformation.streamline.absorb import AbsorbConsecutiveTransposes
from finn.util.exception import FINNUserError
from finn.util.fpgadataflow import is_hls_node, is_rtl_node
from finn.util.logging import log


# NICCHANGE: temporary. models is wrong.
def _fix_mvau_weight_dtype(model: ModelWrapper):
    # sets MVAU weightDataType to INT4 (which is correct)
    # it is showing up as INT64 for some reason, but we know its INT4
    for node in model.graph.node:
        if "MVAU" in node.op_type:
            inst = getCustomOp(node)
            if node.name == "MVAU_12":
                inst.set_nodeattr("weightDataType", "INT8")
            else:
                inst.set_nodeattr("weightDataType", "INT4")
        elif node.op_type == "FINNLoop":
            loop_model = getCustomOp(node).get_nodeattr("body")
            _fix_mvau_weight_dtype(loop_model)
            getCustomOp(node).set_nodeattr("body", loop_model.graph)
    return model


def step_fix_mvau_weight_dtype(model: ModelWrapper, cfg: DataflowBuildConfig):
    return _fix_mvau_weight_dtype(model)


def step_convert_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig):
    # Start with inferring splitting and concatenating infrastructure operators
    # as these also address QONNX type inference defects.
    model = model.transform(hardware.InferSplitLayer())
    model = model.transform(hardware.InferConcatLayer())

    # Convert pooling and convolution-related operators next as these might
    # affect type and shape inference for subsequent transformations.
    model = model.transform(hardware.InferPool())
    model = model.transform(hardware.InferPoolFromReduce())
    model = model.transform(hardware.InferConvInpGen())
    model = model.transform(hardware.InferFMPadding())

    # Infer fused scaled dot-product attention before inferring standalone
    # thresholds and MVUs
    model = model.transform(InferScaledDotProductAttention())

    # Usual FINN hardware conversion transformations: Start with standalone
    # thresholds to avoid fusing these into MVAUs, from there on the order does
    # not really matter.
    if cfg.standalone_thresholds:
        model = model.transform(hardware.InferThresholdingLayer())

    model = model.transform(hardware.InferBinaryMatrixVectorActivation())
    model = model.transform(hardware.InferQuantizedMatrixVectorActivation())
    model = model.transform(hardware.InferVectorVectorActivation())
    model = model.transform(hardware.InferThresholdingLayer())
    model = model.transform(hardware.InferLabelSelectLayer())
    model = model.transform(hardware.InferLabelSelectLayer())

    # Inferring Gather as Lookup layers needs some special treatment: The ONNX
    # standard requires signed-integer index inputs whereas FINN assumes float
    # container types annotated as unsigned integer.
    for index, node in enumerate(model.graph.node):
        # If this is a Gather node, force the input (index) type annotation
        if node.op_type == "Gather":
            # Force to unsigned 64-bit integer for now
            model.set_tensor_datatype(node.input[1], DataType["UINT64"])
            # Get the value info for the input tensor to have access to the ONNX
            # datatype of the tensor
            value_info = model.get_tensor_valueinfo(node.input[1])
            # Force the container datatype of the input to be a float
            value_info.type.tensor_type.elem_type = 1

    # Now Gather operators should be inferred as Lookup layers
    model = model.transform(hardware.InferLookupLayer())

    # TODO: Theses two should actually be considered as streamlining of already
    #  lowered operator representations, but pooling conversion above might
    #  introduce new transposes to the graph
    model = model.transform(AbsorbConsecutiveTransposes())
    model = model.transform(RemoveCNVtoFCFlatten())

    # Implement remaining elementwise binary operators as hardware operators,
    # including floating-point operators, excluding the final dequantizer scale.
    model = model.transform(
        hardware.InferElementwiseBinaryOperation(
            hardware.InferElementwiseBinaryOperation.reject_output_dequant
        )
    )
    # Any remaining reshape operator must be implemented to keep the graph valid
    # while also not breaking the chain of FINN operators.
    model = model.transform(hardware.InferReshape())
    # Explicitly replicate stream connections between layers as hardware does
    # not allow multiple consumer of a single AXI stream.
    model = model.transform(InferReplicateStream())

    # Cleanup the graph after hardware conversion by redoing type and shape
    # inference. There is also more potential for rounding thresholds after
    # addressing QONNX type inference defects regarding Split and Concat.
    model = model.transform(InferDataLayouts())
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(RoundAndClipThresholds())

    # TODO: Giving unique node names apparently is not part of the default
    #  cleanup transformations...?
    return model.transform(GiveUniqueNodeNames())


# NICCHANGE:
# HLS caps ap_uint template parameter at 8191 bits. 
# Folding must respect this
_ATTENTION_MAX_STREAM_WIDTH = 8191
def _attention_tile_widths(inst, embfold, seqfold):
    qkdim, _, vdim, kvlen = inst.shapes
    ktype_bits = DataType[inst.get_nodeattr("KType")].bitwidth()
    vtype_bits = DataType[inst.get_nodeattr("VType")].bitwidth()
    i_elems = qkdim // embfold
    o_elems = vdim // embfold
    s_elems = kvlen // seqfold
    return ktype_bits * i_elems * s_elems, vtype_bits * o_elems * s_elems


def _set_folding_attention(model: ModelWrapper, target_cycles_per_frame,
                           max_stream_width=_ATTENTION_MAX_STREAM_WIDTH):
    # Run over all nodes in the model graph to look for attention operators,
    # which are currently not handled by the SetFolding transformation
    for index, node in enumerate(model.graph.node):
        if node.op_type == "ScaledDotProductAttention_hls":
            # Convert this to the custom-op instance for easy access to node
            # attributes
            inst = getCustomOp(node)

            # Extract sequence length and embedding dimension from the
            # operator attributes
            qkdim, qlen, vdim, kvlen = inst.shapes

            # Initialize folding to sequential processing. As the same
            # folding is applied to query, key and value embeddings, but
            # these might be different, use the greatest common divisor.
            inst.set_nodeattr("EmbFold", math.gcd(qkdim, vdim))
            inst.set_nodeattr("SeqFold", kvlen)

            # Try to unfold along the embedding dimension first, increasing
            # parallelism in steps following the common divisors the inputs.
            for fold in reversed(common_divisors([qkdim, vdim])):
                k_w, v_w = _attention_tile_widths(
                    inst, fold, inst.get_nodeattr("SeqFold"))
                if k_w > max_stream_width or v_w > max_stream_width:
                    log.info(
                        f"{node.name}: skipping EmbFold={fold} "
                        f"(K-tile={k_w}b, V-tile={v_w}b > {max_stream_width}b)"
                    )
                    continue
                # Configure the folding attribute
                inst.set_nodeattr("EmbFold", fold)
                # Check if this is sufficient to meet the cycles target
                if inst.get_exp_cycles() <= target_cycles_per_frame:
                    break

            # Try to unfold along the sequence dimension next, increasing
            # parallelism in steps divisors of the key and value sequence.
            for fold in reversed(common_divisors([kvlen])):
                k_w, v_w = _attention_tile_widths(
                    inst, inst.get_nodeattr("EmbFold"), fold)
                if k_w > max_stream_width or v_w > max_stream_width:
                    log.info(
                        f"{node.name}: skipping SeqFold={fold} "
                        f"(K-tile={k_w}b, V-tile={v_w}b > {max_stream_width}b)"
                    )
                    continue
                # Configure the folding attribute
                inst.set_nodeattr("SeqFold", fold)
                # Check if this is sufficient to meet the cycles target
                if inst.get_exp_cycles() <= target_cycles_per_frame:
                    break

    # Annotate cycles estimates for all operators n the model
    model = model.transform(GiveUniqueNodeNames())
    return model.transform(AnnotateCycles())


def _critical_path_node_count(model: ModelWrapper) -> int:
    depth: dict[str, int] = {}
    for node in model.graph.node:
        predecessors = model.find_direct_predecessors(node)
        max_pred = 0 if not predecessors else max(depth.get(p.name, 0) for p in predecessors)
        depth[node.name] = 1 + max_pred
    return max(depth.values()) if depth else 1


def step_set_folding(model: ModelWrapper, cfg: DataflowBuildConfig):
    #log.info(f"two_pass_relaxation = {cfg.folding_two_pass_relaxation}")
    cfg.folding_two_pass_relaxation = False
    # Resolve the target cycles per from the build configuration, considering
    # clock and target throughput
    target_cycles_per_frame = cfg._resolve_cycles_per_frame()

    # If no target_cycles_per_frame is given, assume we are provided with a full
    # folding config JSON (including parallelism settings): Apply it and skip remaining step
    if target_cycles_per_frame is None:
        if cfg.folding_config_file is None:
            raise FINNUserError("Neither FPS target nor folding config provided.")
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(ApplyConfig(cfg.folding_config_file))
        return model

    # Set folding to target cycles for all attention operators in the model
    model = _set_folding_attention(model, target_cycles_per_frame)

    # NICCHANGE: Also set folding of FINNLoop bodies
    for node in model.graph.node:
        if node.op_type == "FINNLoop":
            node_inst = getCustomOp(node)
            iterations = node_inst.get_nodeattr("iteration")
            loop_model = node_inst.get_nodeattr("body")
            k = _critical_path_node_count(loop_model)
            #loop_target = target_cycles_per_frame // (iterations * k)
            loop_target = target_cycles_per_frame // iterations
            log.info(
                f"FINNLoop {node.name}: critical path length k={k}, "
                f"iterations={iterations}, loop-body target cycles={loop_target}"
            )
            loop_model = _set_folding_attention(loop_model, loop_target)
            # TODO: Experiment with two_pass_relaxation for the loop body as well
            loop_model = loop_model.transform(
                SetFolding(loop_target, cfg.mvau_wwidth_max, two_pass_relaxation=False),
            )
            node_inst.set_nodeattr("body", loop_model.graph)

    # Run AnnotateCycles after all FINNLoop's have been folded
    model = model.transform(AnnotateCycles())

    # Use FINN auto-folding to configure all other operators to reach the target cycles. 
    model = model.transform(
        SetFolding(target_cycles_per_frame, cfg.mvau_wwidth_max, 
                   two_pass_relaxation=cfg.folding_two_pass_relaxation),
    )

    perf_dict = model.analysis(dataflow_performance)
    max_cycles = perf_dict["max_cycles"]
    log.info(f"max_cycles: {max_cycles}, target_cycles_per_frame: {target_cycles_per_frame}")
    log.info(perf_dict)

    # NICCHANGE TODO: Do this for the loop body as well?
    # Two-pass relaxation for attention operators: Redo folding settings
    # with lower target based on cycles of the slowest operator
    if cfg.folding_two_pass_relaxation:
        # Extract dataflow performance from the graph (has been annotated by
        # SetFolding transformation)
        perf_dict = model.analysis(dataflow_performance)
        # Estimated cycles are above the target, so this is bottlenecked by
        # some operator
        if perf_dict["max_cycles"] > target_cycles_per_frame:
            # Set folding to lower target cycles for all attention operators
            # in the model
            model = _set_folding_attention(model, perf_dict["max_cycles"])

    # TODO: The following export of auto_folding_config.yaml is largely redundant
    # because later steps generate a final_hw_config.json, so it may be removed

    # Hardware attributes to be extracted from each node
    hw_attrs = {
        "PE",
        "SIMD",
        "EmbFold",
        "SeqFold",
        "parallel_window",
        "ram_style",
        "ram_style_thresholds",
        "ram_style_mask",
        "depth",
        "impl_style",
        "resType",
        "mac_resource",
        "mem_mode",
        "runtime_writeable_weights",
        "inFIFODepths",
        "outFIFODepths",
        "depth_trigger_uram",
        "depth_trigger_bram",
    }

    # Start collecting the configuration from the model graph as a dictionary
    config = {"defaults": {}}
    # Iterate all nodes in the graph keeping track of the index
    for index, node in enumerate(model.graph.node):
        # Convert this to the custom-op instance for easy access to node
        # attributes
        inst = getCustomOp(node)
        # Prepare the node-specific configuration entry for this node
        config[node.name] = {}
        # Collect attribute values for all specified hardware attributes
        for key in hw_attrs:
            # Some hardware attributes may not be present for all nodes or
            # op-types, this will be signaled via exception
            try:
                # Try extracting the configuration value from the node
                # custom-op instance
                config[node.name][key] = inst.get_nodeattr(key)
            # Missing attributes are signaled via AttributeError
            except AttributeError:
                pass
        # Cleanup: If no attribute is present for this node, there is no
        # need to keep this in the configuration dictionary as there is
        # nothing to be restored later
        if not config[node.name]:
            # Remove the entry form the configuration dictionary
            del config[node.name]

    # Create/Open a YAML file to store the configuration for later reuse
    with open(cfg.output_dir + "/auto_folding_config.yaml", "w") as file:
        # Store the configuration dictionary as YAML code
        yaml.safe_dump(config, file)

    # If a folding configuration file is given, load and parse YAML and apply to
    # all nodes in the model
    # NOTE: This is intended to apply a folding YAML file that defines sensible
    # defaults for resource selection (mem_mode, ram_style, etc.) in addition to the
    # auto folding above, NOT for setting parallelism attributes (SIMD, PE, etc.)
    if cfg.folding_config_file is not None:
        with open(cfg.folding_config_file, "r") as file:
            config = yaml.safe_load(file)

        for index, node in enumerate(model.graph.node):
            # A node should not be named "defaults"...
            assert node.name != "defaults", "Node has reserved name 'defaults'"
            # Convert this to the custom-op instance for easy access to node
            # attributes
            inst = getCustomOp(node)
            # Apply the per operator type default configurations to the node
            if node.op_type in config["defaults"]:
                # Run over all default options to be applied to this node
                for key, value in config["defaults"][node.op_type].items():
                    # Set the nodes attribute to the default option value
                    inst.set_nodeattr(key, value)
            # If there is an individual, node-specific configuration apply
            # this next, potentially overriding the defaults set above
            if node.name in config:
                # Run over all node-specific options to be applied to this
                # node
                for key, value in config[node.name].items():
                    # Set the nodes attribute to the option value
                    inst.set_nodeattr(key, value)

    # Model with applied auto and manual folding configuration
    return model

