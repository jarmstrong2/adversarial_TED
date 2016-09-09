### TIMESTEP MODEL

| Variable          | Value     |
| :---------------- | :---------|
| timesteps         | 16         |
| lstm_layers_RNN_g | 6        |
| lstm_layers_RNN_d | 2         |
| hidden_size_RNN_g | 600       |
| hidden_size_RNN_d | 400       |
| lr                | 2e-4:GEN/1e-4:DISC    |
| iterations        | > 5e5       |

#### SAMPLES

|0|1|2|3|4|5|6|7|8|9|
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|![alt tag](steps16/im0_0.png)|![alt tag](steps16/im1_0.png)|![alt tag](steps16/im2_0.png)|![alt tag](steps16/im3_0.png)|![alt tag](steps16/im4_0.png)|![alt tag](steps16/im5_0.png)|![alt tag](steps16/im6_0.png)|![alt tag](steps16/im7_0.png)|![alt tag](steps16/im8_0.png)|![alt tag](steps16/im9_0.png)|
|![alt tag](steps16/im0_1.png)|![alt tag](steps16/im1_1.png)|![alt tag](steps16/im2_1.png)|![alt tag](steps16/im3_1.png)|![alt tag](steps16/im4_1.png)|![alt tag](steps16/im5_1.png)|![alt tag](steps16/im6_1.png)|![alt tag](steps16/im7_1.png)|![alt tag](steps16/im8_1.png)|![alt tag](steps16/im9_1.png)|
|![alt tag](steps16/im0_2.png)|![alt tag](steps16/im1_2.png)|![alt tag](steps16/im2_2.png)|![alt tag](steps16/im3_2.png)|![alt tag](steps16/im4_2.png)|![alt tag](steps16/im5_2.png)|![alt tag](steps16/im6_2.png)|![alt tag](steps16/im7_2.png)|![alt tag](steps16/im8_2.png)|![alt tag](steps16/im9_2.png)|
|![alt tag](steps16/im0_3.png)|![alt tag](steps16/im1_3.png)|![alt tag](steps16/im2_3.png)|![alt tag](steps16/im3_3.png)|![alt tag](steps16/im4_3.png)|![alt tag](steps16/im5_3.png)|![alt tag](steps16/im6_3.png)|![alt tag](steps16/im7_3.png)|![alt tag](steps16/im8_3.png)|![alt tag](steps16/im9_3.png)|
|![alt tag](steps16/im0_4.png)|![alt tag](steps16/im1_4.png)|![alt tag](steps16/im2_4.png)|![alt tag](steps16/im3_4.png)|![alt tag](steps16/im4_4.png)|![alt tag](steps16/im5_4.png)|![alt tag](steps16/im6_4.png)|![alt tag](steps16/im7_4.png)|![alt tag](steps16/im8_4.png)|![alt tag](steps16/im9_4.png)|
|![alt tag](steps16/im0_5.png)|![alt tag](steps16/im1_5.png)|![alt tag](steps16/im2_5.png)|![alt tag](steps16/im3_5.png)|![alt tag](steps16/im4_5.png)|![alt tag](steps16/im5_5.png)|![alt tag](steps16/im6_5.png)|![alt tag](steps16/im7_5.png)|![alt tag](steps16/im8_5.png)|![alt tag](steps16/im9_5.png)|
|![alt tag](steps16/im0_6.png)|![alt tag](steps16/im1_6.png)|![alt tag](steps16/im2_6.png)|![alt tag](steps16/im3_6.png)|![alt tag](steps16/im4_6.png)|![alt tag](steps16/im5_6.png)|![alt tag](steps16/im6_6.png)|![alt tag](steps16/im7_6.png)|![alt tag](steps16/im8_6.png)|![alt tag](steps16/im9_6.png)|
|![alt tag](steps16/im0_7.png)|![alt tag](steps16/im1_7.png)|![alt tag](steps16/im2_7.png)|![alt tag](steps16/im3_7.png)|![alt tag](steps16/im4_7.png)|![alt tag](steps16/im5_7.png)|![alt tag](steps16/im6_7.png)|![alt tag](steps16/im7_7.png)|![alt tag](steps16/im8_7.png)|![alt tag](steps16/im9_7.png)|
|![alt tag](steps16/im0_8.png)|![alt tag](steps16/im1_8.png)|![alt tag](steps16/im2_8.png)|![alt tag](steps16/im3_8.png)|![alt tag](steps16/im4_8.png)|![alt tag](steps16/im5_8.png)|![alt tag](steps16/im6_8.png)|![alt tag](steps16/im7_8.png)|![alt tag](steps16/im8_8.png)|![alt tag](steps16/im9_8.png)|
|![alt tag](steps16/im0_9.png)|![alt tag](steps16/im1_9.png)|![alt tag](steps16/im2_9.png)|![alt tag](steps16/im3_9.png)|![alt tag](steps16/im4_9.png)|![alt tag](steps16/im5_9.png)|![alt tag](steps16/im6_9.png)|![alt tag](steps16/im7_9.png)|![alt tag](steps16/im8_9.png)|![alt tag](steps16/im9_9.png)|

![alt tag](steps16/loss_sep_4_18.png)

![alt tag](steps16/classification_sep_4_18.png)

![alt tag](steps16/avg_sep_4_18.png)

### TIMESTEP ANNEALED MODEL

| Variable          | Value     |
| :---------------- | :---------|
| timesteps         | 4         |
| lstm_layers_RNN_g | 6        |
| lstm_layers_RNN_d | 2         |
| hidden_size_RNN_g | 600       |
| hidden_size_RNN_d | 400       |
| lr                | 2e-4:GEN/1e-4:DISC    |
| anneal schedule                | after 1e6 iterations -- learning_rate - (3e-8 * ((self.global_step) / 1000.0)):GEN/learning_rate - (1.33e-8 * ((self.global_step) / 1000.0)):DISC    |
| iterations        | > 2.5e6       |

#### SAMPLES

|0|1|2|3|4|5|6|7|8|9|
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|![alt tag](ann_ims/im0_0.png)|![alt tag](ann_ims/im1_0.png)|![alt tag](ann_ims/im2_0.png)|![alt tag](ann_ims/im3_0.png)|![alt tag](ann_ims/im4_0.png)|![alt tag](ann_ims/im5_0.png)|![alt tag](ann_ims/im6_0.png)|![alt tag](ann_ims/im7_0.png)|![alt tag](ann_ims/im8_0.png)|![alt tag](ann_ims/im9_0.png)|
|![alt tag](ann_ims/im0_1.png)|![alt tag](ann_ims/im1_1.png)|![alt tag](ann_ims/im2_1.png)|![alt tag](ann_ims/im3_1.png)|![alt tag](ann_ims/im4_1.png)|![alt tag](ann_ims/im5_1.png)|![alt tag](ann_ims/im6_1.png)|![alt tag](ann_ims/im7_1.png)|![alt tag](ann_ims/im8_1.png)|![alt tag](ann_ims/im9_1.png)|
|![alt tag](ann_ims/im0_2.png)|![alt tag](ann_ims/im1_2.png)|![alt tag](ann_ims/im2_2.png)|![alt tag](ann_ims/im3_2.png)|![alt tag](ann_ims/im4_2.png)|![alt tag](ann_ims/im5_2.png)|![alt tag](ann_ims/im6_2.png)|![alt tag](full_mod_aug_8/im7_2.png)|![alt tag](ann_ims/im8_2.png)|![alt tag](ann_ims/im9_2.png)|
|![alt tag](ann_ims/im0_3.png)|![alt tag](ann_ims/im1_3.png)|![alt tag](ann_ims/im2_3.png)|![alt tag](ann_ims/im3_3.png)|![alt tag](ann_ims/im4_3.png)|![alt tag](ann_ims/im5_3.png)|![alt tag](ann_ims/im6_3.png)|![alt tag](ann_ims/im7_3.png)|![alt tag](ann_ims/im8_3.png)|![alt tag](ann_ims/im9_3.png)|
|![alt tag](ann_ims/im0_4.png)|![alt tag](ann_ims/im1_4.png)|![alt tag](ann_ims/im2_4.png)|![alt tag](ann_ims/im3_4.png)|![alt tag](ann_ims/im4_4.png)|![alt tag](ann_ims/im5_4.png)|![alt tag](ann_ims/im6_4.png)|![alt tag](ann_ims/im7_4.png)|![alt tag](ann_ims/im8_4.png)|![alt tag](ann_ims/im9_4.png)|
|![alt tag](ann_ims/im0_5.png)|![alt tag](ann_ims/im1_5.png)|![alt tag](ann_ims/im2_5.png)|![alt tag](ann_ims/im3_5.png)|![alt tag](ann_ims/im4_5.png)|![alt tag](ann_ims/im5_5.png)|![alt tag](ann_ims/im6_5.png)|![alt tag](ann_ims/im7_5.png)|![alt tag](ann_ims/im8_5.png)|![alt tag](ann_ims/im9_5.png)|
|![alt tag](ann_ims/im0_6.png)|![alt tag](ann_ims/im1_6.png)|![alt tag](ann_ims/im2_6.png)|![alt tag](ann_ims/im3_6.png)|![alt tag](ann_ims/im4_6.png)|![alt tag](ann_ims/im5_6.png)|![alt tag](ann_ims/im6_6.png)|![alt tag](ann_ims/im7_6.png)|![alt tag](ann_ims/im8_6.png)|![alt tag](ann_ims/im9_6.png)|
|![alt tag](ann_ims/im0_7.png)|![alt tag](ann_ims/im1_7.png)|![alt tag](ann_ims/im2_7.png)|![alt tag](ann_ims/im3_7.png)|![alt tag](ann_ims/im4_7.png)|![alt tag](ann_ims/im5_7.png)|![alt tag](ann_ims/im6_7.png)|![alt tag](ann_ims/im7_7.png)|![alt tag](ann_ims/im8_7.png)|![alt tag](ann_ims/im9_7.png)|
|![alt tag](ann_ims/im0_8.png)|![alt tag](ann_ims/im1_8.png)|![alt tag](ann_ims/im2_8.png)|![alt tag](ann_ims/im3_8.png)|![alt tag](ann_ims/im4_8.png)|![alt tag](ann_ims/im5_8.png)|![alt tag](ann_ims/im6_8.png)|![alt tag](ann_ims/im7_8.png)|![alt tag](ann_ims/im8_8.png)|![alt tag](ann_ims/im9_8.png)|
|![alt tag](ann_ims/im0_9.png)|![alt tag](ann_ims/im1_9.png)|![alt tag](ann_ims/im2_9.png)|![alt tag](ann_ims/im3_9.png)|![alt tag](ann_ims/im4_9.png)|![alt tag](ann_ims/im5_9.png)|![alt tag](ann_ims/im6_9.png)|![alt tag](ann_ims/im7_9.png)|![alt tag](ann_ims/im8_9.png)|![alt tag](ann_ims/im9_9.png)|

![alt tag](loss_anneal.png)

![alt tag](classification_anneal.png)

![alt tag](lr_anneal.png)

### TIMESTEP MODEL

| Variable          | Value     |
| :---------------- | :---------|
| timesteps         | 4         |
| lstm_layers_RNN_g | 6        |
| lstm_layers_RNN_d | 2         |
| hidden_size_RNN_g | 600       |
| hidden_size_RNN_d | 400       |
| lr                | 1e-4    |
| iterations        | > 2.5e6       |

#### SAMPLES
![drawing](faster_transition_smaller.gif)

|0|1|2|3|4|5|6|7|8|9|
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|![alt tag](full_mod_aug_8/im0_1.png)|![alt tag](full_mod_aug_8/im1_1.png)|![alt tag](full_mod_aug_8/im2_1.png)|![alt tag](full_mod_aug_8/im3_1.png)|![alt tag](full_mod_aug_8/im4_1.png)|![alt tag](full_mod_aug_8/im5_1.png)|![alt tag](full_mod_aug_8/im6_1.png)|![alt tag](full_mod_aug_8/im7_1.png)|![alt tag](full_mod_aug_8/im8_1.png)|![alt tag](full_mod_aug_8/im9_1.png)|
|![alt tag](full_mod_aug_8/im0_2.png)|![alt tag](full_mod_aug_8/im1_2.png)|![alt tag](full_mod_aug_8/im2_2.png)|![alt tag](full_mod_aug_8/im3_2.png)|![alt tag](full_mod_aug_8/im4_2.png)|![alt tag](full_mod_aug_8/im5_2.png)|![alt tag](full_mod_aug_8/im6_2.png)|![alt tag](full_mod_aug_8/im7_2.png)|![alt tag](full_mod_aug_8/im8_2.png)|![alt tag](full_mod_aug_8/im9_2.png)|
|![alt tag](full_mod_aug_8/im0_3.png)|![alt tag](full_mod_aug_8/im1_3.png)|![alt tag](full_mod_aug_8/im2_3.png)|![alt tag](full_mod_aug_8/im3_3.png)|![alt tag](full_mod_aug_8/im4_3.png)|![alt tag](full_mod_aug_8/im5_3.png)|![alt tag](full_mod_aug_8/im6_3.png)|![alt tag](full_mod_aug_8/im7_3.png)|![alt tag](full_mod_aug_8/im8_3.png)|![alt tag](full_mod_aug_8/im9_3.png)|
|![alt tag](full_mod_aug_8/im0_4.png)|![alt tag](full_mod_aug_8/im1_4.png)|![alt tag](full_mod_aug_8/im2_4.png)|![alt tag](full_mod_aug_8/im3_4.png)|![alt tag](full_mod_aug_8/im4_4.png)|![alt tag](full_mod_aug_8/im5_4.png)|![alt tag](full_mod_aug_8/im6_4.png)|![alt tag](full_mod_aug_8/im7_4.png)|![alt tag](full_mod_aug_8/im8_4.png)|![alt tag](full_mod_aug_8/im9_4.png)|
|![alt tag](full_mod_aug_8/im0_5.png)|![alt tag](full_mod_aug_8/im1_5.png)|![alt tag](full_mod_aug_8/im2_5.png)|![alt tag](full_mod_aug_8/im3_5.png)|![alt tag](full_mod_aug_8/im4_5.png)|![alt tag](full_mod_aug_8/im5_5.png)|![alt tag](full_mod_aug_8/im6_5.png)|![alt tag](full_mod_aug_8/im7_5.png)|![alt tag](full_mod_aug_8/im8_5.png)|![alt tag](full_mod_aug_8/im9_5.png)|
|![alt tag](full_mod_aug_8/im0_6.png)|![alt tag](full_mod_aug_8/im1_6.png)|![alt tag](full_mod_aug_8/im2_6.png)|![alt tag](full_mod_aug_8/im3_6.png)|![alt tag](full_mod_aug_8/im4_6.png)|![alt tag](full_mod_aug_8/im5_6.png)|![alt tag](full_mod_aug_8/im6_6.png)|![alt tag](full_mod_aug_8/im7_6.png)|![alt tag](full_mod_aug_8/im8_6.png)|![alt tag](full_mod_aug_8/im9_6.png)|
|![alt tag](full_mod_aug_8/im0_7.png)|![alt tag](full_mod_aug_8/im1_7.png)|![alt tag](full_mod_aug_8/im2_7.png)|![alt tag](full_mod_aug_8/im3_7.png)|![alt tag](full_mod_aug_8/im4_7.png)|![alt tag](full_mod_aug_8/im5_7.png)|![alt tag](full_mod_aug_8/im6_7.png)|![alt tag](full_mod_aug_8/im7_7.png)|![alt tag](full_mod_aug_8/im8_7.png)|![alt tag](full_mod_aug_8/im9_7.png)|
|![alt tag](full_mod_aug_8/im0_8.png)|![alt tag](full_mod_aug_8/im1_8.png)|![alt tag](full_mod_aug_8/im2_8.png)|![alt tag](full_mod_aug_8/im3_8.png)|![alt tag](full_mod_aug_8/im4_8.png)|![alt tag](full_mod_aug_8/im5_8.png)|![alt tag](full_mod_aug_8/im6_8.png)|![alt tag](full_mod_aug_8/im7_8.png)|![alt tag](full_mod_aug_8/im8_8.png)|![alt tag](full_mod_aug_8/im9_8.png)|
|![alt tag](full_mod_aug_8/im0_9.png)|![alt tag](full_mod_aug_8/im1_9.png)|![alt tag](full_mod_aug_8/im2_9.png)|![alt tag](full_mod_aug_8/im3_9.png)|![alt tag](full_mod_aug_8/im4_9.png)|![alt tag](full_mod_aug_8/im5_9.png)|![alt tag](full_mod_aug_8/im6_9.png)|![alt tag](full_mod_aug_8/im7_9.png)|![alt tag](full_mod_aug_8/im8_9.png)|![alt tag](full_mod_aug_8/im9_9.png)|
|![alt tag](full_mod_aug_8/im0_10.png)|![alt tag](full_mod_aug_8/im1_10.png)|![alt tag](full_mod_aug_8/im2_10.png)|![alt tag](full_mod_aug_8/im3_10.png)|![alt tag](full_mod_aug_8/im4_10.png)|![alt tag](full_mod_aug_8/im5_10.png)|![alt tag](full_mod_aug_8/im6_10.png)|![alt tag](full_mod_aug_8/im7_10.png)|![alt tag](full_mod_aug_8/im8_10.png)|![alt tag](full_mod_aug_8/im9_10.png)|

![alt tag](loss_full_aug_8.png)

![alt tag](classification_full_aug_8.png)

### SANITY (FULL IMAGE) MODEL

| Variable          | Value     |
| :---------------- | :---------|
| timesteps         | 1         |
| lstm_layers_RNN_g | 5        |
| lstm_layers_RNN_d | 2         |
| hidden_size_RNN_g | 600       |
| hidden_size_RNN_d | 400       |
| lr                | 1e-4    |
| iterations        | 2e6       |
| classification (discriminator)        | 0.609      |

#### SAMPLES

|0|1|2|3|4|5|6|7|8|9|
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|![alt tag](samples/im0_sanity_1.png)|![alt tag](samples/im1_sanity_1.png)|![alt tag](samples/im2_sanity_1.png)|![alt tag](samples/im3_sanity_1.png)|![alt tag](samples/im4_sanity_1.png)|![alt tag](samples/im5_sanity_1.png)|![alt tag](samples/im6_sanity_1.png)|![alt tag](samples/im7_sanity_1.png)|![alt tag](samples/im8_sanity_1.png)|![alt tag](samples/im9_sanity_1.png)|
|![alt tag](samples/im0_sanity_2.png)|![alt tag](samples/im1_sanity_2.png)|![alt tag](samples/im2_sanity_2.png)|![alt tag](samples/im3_sanity_2.png)|![alt tag](samples/im4_sanity_2.png)|![alt tag](samples/im5_sanity_2.png)|![alt tag](samples/im6_sanity_2.png)|![alt tag](samples/im7_sanity_2.png)|![alt tag](samples/im8_sanity_2.png)|![alt tag](samples/im9_sanity_2.png)|
|![alt tag](samples/im0_sanity_3.png)|![alt tag](samples/im1_sanity_3.png)|![alt tag](samples/im2_sanity_3.png)|![alt tag](samples/im3_sanity_3.png)|![alt tag](samples/im4_sanity_3.png)|![alt tag](samples/im5_sanity_3.png)|![alt tag](samples/im6_sanity_3.png)|![alt tag](samples/im7_sanity_3.png)|![alt tag](samples/im8_sanity_3.png)|![alt tag](samples/im9_sanity_3.png)|
|![alt tag](samples/im0_sanity_4.png)|![alt tag](samples/im1_sanity_4.png)|![alt tag](samples/im2_sanity_4.png)|![alt tag](samples/im3_sanity_4.png)|![alt tag](samples/im4_sanity_4.png)|![alt tag](samples/im5_sanity_4.png)|![alt tag](samples/im6_sanity_4.png)|![alt tag](samples/im7_sanity_4.png)|![alt tag](samples/im8_sanity_4.png)|![alt tag](samples/im9_sanity_4.png)|
|![alt tag](samples/im0_sanity_5.png)|![alt tag](samples/im1_sanity_5.png)|![alt tag](samples/im2_sanity_5.png)|![alt tag](samples/im3_sanity_5.png)|![alt tag](samples/im4_sanity_5.png)|![alt tag](samples/im5_sanity_5.png)|![alt tag](samples/im6_sanity_5.png)|![alt tag](samples/im7_sanity_5.png)|![alt tag](samples/im8_sanity_5.png)|![alt tag](samples/im9_sanity_5.png)|
|![alt tag](samples/im0_sanity_6.png)|![alt tag](samples/im1_sanity_6.png)|![alt tag](samples/im2_sanity_6.png)|![alt tag](samples/im3_sanity_6.png)|![alt tag](samples/im4_sanity_6.png)|![alt tag](samples/im5_sanity_6.png)|![alt tag](samples/im6_sanity_6.png)|![alt tag](samples/im7_sanity_6.png)|![alt tag](samples/im8_sanity_6.png)|![alt tag](samples/im9_sanity_6.png)|
|![alt tag](samples/im0_sanity_7.png)|![alt tag](samples/im1_sanity_7.png)|![alt tag](samples/im2_sanity_7.png)|![alt tag](samples/im3_sanity_7.png)|![alt tag](samples/im4_sanity_7.png)|![alt tag](samples/im5_sanity_7.png)|![alt tag](samples/im6_sanity_7.png)|![alt tag](samples/im7_sanity_7.png)|![alt tag](samples/im8_sanity_7.png)|![alt tag](samples/im9_sanity_7.png)|
|![alt tag](samples/im0_sanity_8.png)|![alt tag](samples/im1_sanity_8.png)|![alt tag](samples/im2_sanity_8.png)|![alt tag](samples/im3_sanity_8.png)|![alt tag](samples/im4_sanity_8.png)|![alt tag](samples/im5_sanity_8.png)|![alt tag](samples/im6_sanity_8.png)|![alt tag](samples/im7_sanity_8.png)|![alt tag](samples/im8_sanity_8.png)|![alt tag](samples/im9_sanity_8.png)|
|![alt tag](samples/im0_sanity_9.png)|![alt tag](samples/im1_sanity_9.png)|![alt tag](samples/im2_sanity_9.png)|![alt tag](samples/im3_sanity_9.png)|![alt tag](samples/im4_sanity_9.png)|![alt tag](samples/im5_sanity_9.png)|![alt tag](samples/im6_sanity_9.png)|![alt tag](samples/im7_sanity_9.png)|![alt tag](samples/im8_sanity_9.png)|![alt tag](samples/im9_sanity_9.png)|
|![alt tag](samples/im0_sanity_10.png)|![alt tag](samples/im1_sanity_10.png)|![alt tag](samples/im2_sanity_10.png)|![alt tag](samples/im3_sanity_10.png)|![alt tag](samples/im4_sanity_10.png)|![alt tag](samples/im5_sanity_10.png)|![alt tag](samples/im6_sanity_10.png)|![alt tag](samples/im7_sanity_10.png)|![alt tag](samples/im8_sanity_10.png)|![alt tag](samples/im9_sanity_10.png)|

#### LOSS (Generator)

![alt tag](loss.png)

#### CLASSIFICATION (Generator)

![alt tag](classification.png)

