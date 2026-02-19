# Terramind Segmentation IOProcessor Plugin

This plugin targets segmentation tasks for Terramind models and allows for
multimodal input data (e.g., DEM, optical imagery) to be provided via URLs or
file paths, organized in separate directories by modality.

During initialization, the plugin accesses the model's data module configuration
from the vLLM configuration and instantiates a DataModule object dynamically.

This plugin is installed as `terratorch_tm_segmentation`.

## Plugin specification

### Model requirements

This plugin expects the model to take parameters for inference. The primary
parameter, named `pixel_values`, points to a tensor containing the raw image
data extracted from the input files. Additional parameters like
`location_coords` may be optional depending on the model configuration.

Below an example input model specification accepted by this plugin. The user can
change the shapes of the tensors according to their model requirements but the
number and names of the fields must be kept unchanged.

```json title="Model input specification accepted by the Terramind Segmentation IOProcessor plugin"
"input":{
    "target": "pixel_values",
    "data":{
        "pixel_values":{
            "type": "torch.Tensor",
            "shape": [6, 512, 512]
        },
        "location_coords":{
            "type":"torch.Tensor",
            "shape": [1, 2]
        }
    }
}
```

Full details on TerraTorch models input model specification for vLLM are
available [here](../prepare_your_model.md#model-input-specification).

### Plugin configuration

This plugin allows for additional configuration data to be passed via the
`TERRATORCH_SEGMENTATION_IO_PROCESSOR_CONFIG` environment variable. If set, the
variable should contain the plugin configuration in json string format.

The plugin configuration format is defined in the `PluginConfig` class.

:::terratorch.vllm.plugins.segmentation.types.PluginConfig

### Request Data Format

The input format for the plugin is defined in the `RequestData` class.

:::terratorch.vllm.plugins.segmentation.types.RequestData

For this plugin, the `data` field should contain a dictionary where keys are
modality names (e.g., "DEM", "optical") and values are URLs or file paths
pointing to the respective data files.

Depending on the values set in `data_format`, the plugin expects `data` to
contain strings that comply with the format. Similarly, `out_data_format`
controls the data format returned to the user.

The optional `out_path` field allows you to specify a custom output directory
for the generated GeoTiff file when `out_data_format` is set to `"path"`. If
`out_path` is not provided, the plugin will use the default output path from the
plugin configuration (set via the `TERRATORCH_SEGMENTATION_IO_PROCESSOR_CONFIG`
environment variable).

**Example request payload with URL input and base64 output:**

```json
{
  "data_format": "url",
  "out_data_format": "b64_json",
  "data": {
    "DEM": "https://example.com/path/to/dem_file.tif",
    "optical": "https://example.com/path/to/optical_file.tif"
  }
}
```

**Example request payload with path output and custom output directory:**

```json
{
  "data_format": "url",
  "out_data_format": "path",
  "out_path": "/custom/output/directory",
  "data": {
    "DEM": "https://example.com/path/to/dem_file.tif",
    "optical": "https://example.com/path/to/optical_file.tif"
  }
}
```

### Request Output Format

The output format for the plugin is defined in the `RequestOutput` class.

:::terratorch.vllm.plugins.segmentation.types.RequestOutput

### Plugin Defaults

#### Tiled Inference Parameters

By default the plugin uses the same horizontal and vertical crop value of 512
when computing image tiles. Users can use different crop values by specifying
them in their model `config.json` file. See the example below that overrides the
default values with vertical and horizontal crop values of 256.

```json title="Custom tiled inference parameters in model configuration"
{
  "pretrained_cfg": {
    "model": {
      "init_args": {
        "tiled_inference_parameters": {
          "h_crop": 256,
          "w_crop": 256,
          "delta": 8
        }
      }
    }
  }
}
```

Please note, the `tiled_inference_parameters` field is not mandatory in the
model configuration. Full details on the model configuration file can be found
[here](../prepare_your_model.md#vllm-compatible-model-configuration).

#### Multimodal Data Organization

The plugin expects input data to be organized by modality. When using URL-based
input (`data_format: "url"`), the plugin will automatically download files and
organize them into directories named after each modality.

For example, if your request includes:

```json
{
  "data": {
    "DEM": "https://example.com/dem_file.tif",
    "optical": "https://example.com/optical_file.tif"
  }
}
```

The plugin will create:

- A directory named "DEM" containing the downloaded DEM file
- A directory named "optical" containing the downloaded optical file

When using path-based input (`data_format: "path"`), provide the root directory
path that already contains the modality subdirectories organized in the same
structure.

#### DataModule Configuration

The plugin dynamically instantiates a DataModule based on the configuration in
the model's `config.json` file. For ImpactMeshDataModule, the plugin
automatically adjusts the `label_grep` parameter to allow flexible input data
organization.
