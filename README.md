# Mangrove-Segmentation

## Project Overview
Mangroves, along with seagrass beds, coral reefs, and tidal marshes, provide essential ecological services but are often undervalued in environmental conservation efforts. The primary challenges include connecting marine conservation project initiators, such as Eversea, with investors and accurately assessing the real-world impact of these conservation and restoration initiatives. Our project, in partnership with TNP and Eversea, aims to address this challenge by developing a quantitative indicator for the evolution of marine ecosystems using remote sensing techniques.

## Objective
The objective of our project is to bridge the understanding of marine ecosystems’ crucial roles with technological advances in their conservation, addressing the need for precise data to guide conservation efforts. We introduce an innovative approach using advanced deep learning models for improved coastal organisms segmentation and monitoring. 

## Contributions
- **Dataset Construction**: We constructed and released a new dataset (MagSet-2) that incorporates a variety of spectral bands from the Sentinel-2 satellite as well as mangrove location masks based on the Global Mangrove Watch (GMW). Unlike previous studies, our dataset encompasses all mangrove regions worldwide, accounting for their different species.
  
- **Mamba-Type Models Integration**: We introduced the integration of Mamba-type models into the field of mangrove segmentation, marking the first attempt to apply these models specifically for mangrove segmentation.
  
- **Deep Learning Architectures Comparison**: We compared six leading deep learning architectures—U-Net, MANet, PAN, BEiT, Segformer, and Swin-UMamba—in segmenting mangroves from satellite imagery, representing the forefront of recent advancements in computer vision.

## Code Tutorial

In the UNET.ipynb, you will be able to run the UNET model. By modifying the model definition in the respective cell of the notebook, you can also train the MANet, PAN, BEiT, SegFormer and Swin-UMamba models.

- **PAN**
```{python}
model = smp.PAN(
    encoder_output_stride=16,
    upsampling=4,
    encoder_name = encoder_name,
    #decoder_channels=decoder_channels,
    in_channels=num_channels,
    decoder_channels=512,
    activation = activation,
    classes=1,
)


```
- **MAnet**
```{python}
model = smp.MAnet(
    encoder_name = encoder_name,
    encoder_depth = encoder_depth,
    decoder_channels=decoder_channels,
    in_channels=num_channels,
    activation = activation,
    classes=1,
)
```

- **BEiT**
```{python}

class BEiT(nn.Module):
  def __init__(self):
    super(BEiT,self).__init__()


    configuration = BeitConfig(
        num_labels = 1,
        vocab_size=8192,
        hidden_size=264,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3172,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=128,
        patch_size=16,
        num_channels=9,
        use_mask_token=False,
        use_absolute_position_embeddings=False,
        use_relative_position_bias=False,
        use_shared_relative_position_bias=False,
        layer_scale_init_value=0.1,
        drop_path_rate=0.1,
        use_mean_pooling=True,
        pool_scales=[1, 2, 3, 6],
        use_auxiliary_head=True,
        auxiliary_loss_weight=0.4,
        auxiliary_channels=256,
        auxiliary_num_convs=1,
        auxiliary_concat_input=False,
        semantic_loss_ignore_index=255,
        out_features=None,
        out_indices=[3, 5, 7, 11],
        add_fpn=False,
        reshape_hidden_states=True,
    )

    self.transformer = BeitForSemanticSegmentation(configuration)
    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU()
    self.upsample = nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1)

  def forward(self, images, output_size = (128,128)):
    images = self.transformer(images).logits
    images = self.upsample(images, output_size = (output_size[0]//2,output_size[1]//2))
    images = self.relu(images)
    images = self.upsample(images, output_size = output_size)
    images = self.sigmoid(images)
    return images

model = BEiT()

```
- **SegFormer**
```{python}
class SegFormer(nn.Module):
  def __init__(self):
    super(SegFormer,self).__init__()


    configuration = SegformerConfig(
        num_channels = 9,
        num_labels = 1,
        num_encoder_blocks = 4,
        depths = [3, 3, 3, 3],
        sr_ratios = [8, 4, 2, 1],
        hidden_sizes = [256, 256, 256, 256],
        patch_sizes = [7, 3, 3, 3],
        strides = [4, 2, 2, 2],
        num_attention_heads = [8, 8, 8, 8],
        mlp_ratios = [8, 8, 8, 8],
        hidden_act = 'gelu',
        hidden_dropout_prob = 0.0,
        attention_probs_dropout_prob = 0.0,
        classifier_dropout_prob = 0.1,
        initializer_range = 0.02,
        drop_path_rate = 0.1,
        layer_norm_eps = 1e-06,
        decoder_hidden_size = 128)

    self.transformer = SegformerForSemanticSegmentation(configuration)
    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU()
    self.upsample = nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1)

  def forward(self, images, output_size = (128,128)):
    images = self.transformer(images).logits
    images = self.upsample(images, output_size = (output_size[0]//2,output_size[1]//2))
    images = self.relu(images)
    images = self.upsample(images, output_size = output_size)
    images = self.sigmoid(images)
    return images

model = SegFormer()
)
```


- **Swin-UMamba**
  The code for the Swin-UMamba model will be soon made available.
