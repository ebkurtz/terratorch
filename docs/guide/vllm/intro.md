# Serving TerraTorch models with vLLM

TerraTorch models can be served using the [vLLM](https://github.com/vllm-project/vllm) serving engine. Currently, only models using the `SemanticSegmentationTask` or `PixelwiseRegressionTask` tasks can be served with vLLM.

TerraTorch uses a feature in vLLM called IOProcessor plugins, enabling processing and generation of data in any modality (e.g., geoTiff). An IO Processor plugin is required to perform an end-to-end (i.e., TIFF to TIFF) inference using vLLM. In Terratorch we provide pre-defined IOProcessor plugins, check the list [here](./vllm_io_plugins.md#available-terratorch-ioprocessor-plugins).

To enable your model to be served via vLLM, follow the below steps:

1. Verify the model you want to serve is either already a core model, or learn how to [add your model to TerraTorch](../models.md#adding-a-new-model).
2. [Prepare your model for serving with vLLM](./prepare_your_model.md).
3. [Learn about IOProcessor plugins](./vllm_io_plugins.md), identify an existing one suiting your model or [build one yourself](https://docs.vllm.ai/en/latest/design/io_processor_plugins/).
4. [Start a vLLM serving instance that loads your model](./serving_a_model.md)
