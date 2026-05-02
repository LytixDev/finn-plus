from qonnx.transformation.base import Transformation

from finn.util.logging import log


class NormalizeReplicateStreamOutputs(Transformation):
    """Normalize ReplicateStream output ordering so that the skip/residual
    connection is always output[0]. 

    Without this pass, MLO fails.

    Very ad-hoc. Ideally I fix the exporting script.
    """

    def apply(self, model):
        graph = model.graph

        for node in graph.node:
            if node.op_type != "ReplicateStream_hls":
                continue
            if len(node.output) != 2:
                continue

            out0_is_skip = self._is_skip_output(model, node.output[0])
            out1_is_skip = self._is_skip_output(model, node.output[1])

            if out1_is_skip and not out0_is_skip:
                log.info(
                    f"NormalizeReplicateStreamOutputs: swapping outputs of {node.name} "
                    f"so skip/residual is output[0]"
                )
                node.output[0], node.output[1] = node.output[1], node.output[0]

        return model, False

    @staticmethod
    def _is_skip_output(model, output_name):
        """Check if this output is the skip/residual path.

        The skip path follows: ReplicateStream -> Thresholding -> ElementwiseAdd.
        Returns True if the consumer chain matches this short pattern.
        """
        consumer = model.find_consumer(output_name)
        if consumer is None:
            return False
        if "Thresholding" not in consumer.op_type:
            return False
        if len(consumer.output) == 0:
            return False
        next_consumer = model.find_consumer(consumer.output[0])
        if next_consumer is None:
            return False
        return "ElementwiseAdd" in next_consumer.op_type
