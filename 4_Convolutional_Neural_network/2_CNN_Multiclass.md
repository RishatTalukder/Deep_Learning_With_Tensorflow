# CNN Multi-class Classification

We can follow the same steps we followed in the previous sections:

- Load the Data.
- Preprocess the data.
- Create a model.
- Fit the model.
- Evaluate.
- Try to improve the model by fine tuning different parts of it.


So, first loading the data.

Just like before I've already made a split of the `food 101` dataset and randomly selected 10 classes. You can run the following code to download the whole zip and decompress it.


```python
# !wget "https://github.com/RishatTalukder/Deep_Learning_With_Tensorflow/raw/refs/heads/main/4_Convolutional_Neural_network/resource/10FoodClasses.zip"

# import zipfile

# zip_ref = zipfile.ZipFile('10FoodClasses.zip')
# zip_ref.extractall(path='resource')
# zip_ref.close()
```

Now Let's take a look at the `10FoodClasses` folder like before


```python
import os

for dirpath, dirnames, filenames in os.walk('resource/10FoodClasses'):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
```

    There are 2 directories and 0 images in 'resource/10FoodClasses'.
    There are 10 directories and 0 images in 'resource/10FoodClasses/train'.
    There are 0 directories and 250 images in 'resource/10FoodClasses/train/chicken_wings'.
    There are 0 directories and 250 images in 'resource/10FoodClasses/train/tiramisu'.
    There are 0 directories and 250 images in 'resource/10FoodClasses/train/hamburger'.
    There are 0 directories and 250 images in 'resource/10FoodClasses/train/mussels'.
    There are 0 directories and 250 images in 'resource/10FoodClasses/train/cannoli'.
    There are 0 directories and 250 images in 'resource/10FoodClasses/train/chocolate_cake'.
    There are 0 directories and 250 images in 'resource/10FoodClasses/train/bread_pudding'.
    There are 0 directories and 250 images in 'resource/10FoodClasses/train/ramen'.
    There are 0 directories and 250 images in 'resource/10FoodClasses/train/miso_soup'.
    There are 0 directories and 250 images in 'resource/10FoodClasses/train/creme_brulee'.
    There are 10 directories and 0 images in 'resource/10FoodClasses/val'.
    There are 0 directories and 50 images in 'resource/10FoodClasses/val/chicken_wings'.
    There are 0 directories and 50 images in 'resource/10FoodClasses/val/tiramisu'.
    There are 0 directories and 50 images in 'resource/10FoodClasses/val/hamburger'.
    There are 0 directories and 50 images in 'resource/10FoodClasses/val/mussels'.
    There are 0 directories and 50 images in 'resource/10FoodClasses/val/cannoli'.
    There are 0 directories and 50 images in 'resource/10FoodClasses/val/chocolate_cake'.
    There are 0 directories and 50 images in 'resource/10FoodClasses/val/bread_pudding'.
    There are 0 directories and 50 images in 'resource/10FoodClasses/val/ramen'.
    There are 0 directories and 50 images in 'resource/10FoodClasses/val/miso_soup'.
    There are 0 directories and 50 images in 'resource/10FoodClasses/val/creme_brulee'.


Alright! 800 images in each food class in the training set and 200 images in each food class in the validation set. 

You might ask where is the testing set.

It's the same thing, to be honest. Validation set is used to see performance on the unseen data while training the model and testing set is used to see performance on the unseen data after the model is trained.

Both are unseen datasets. So, We can use it interchangeably.

I'll be using `val` set for both validation and testing.

So, let's load the data.


```python
from keras.utils import image_dataset_from_directory    

training_set = image_dataset_from_directory(
    directory='resource/10FoodClasses/train',
    batch_size=24,
    image_size=(224, 224),
    shuffle=True,
    seed=42
)

validation_set = image_dataset_from_directory(
    directory='resource/10FoodClasses/val',
    batch_size=24,
    image_size=(224, 224),
    shuffle=False,
)
```

    2026-10-25 15:59:50.773290: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


    Found 2500 files belonging to 10 classes.


    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1792922392.422969  214433 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 9493 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3060, pci bus id: 0000:05:00.0, compute capability: 8.6


    Found 500 files belonging to 10 classes.


Wow, now we have 10000 images,(I just hope my pc stays alive). 

So, let's do this.

What classes do we have?


```python
training_set.class_names
```




    ['bread_pudding',
     'cannoli',
     'chicken_wings',
     'chocolate_cake',
     'creme_brulee',
     'hamburger',
     'miso_soup',
     'mussels',
     'ramen',
     'tiramisu']



So, nice diverse classes.

So, let's make a model. This time we have to keep in my our model have to predict `10 classes` instead of just 2. So, the model outpput layer will be a little different.

If you don't know about multi-class classification, you can read one of the previous [article on multi-class classification](https://github.com/RishatTalukder/Deep_Learning_With_Tensorflow/blob/main/pdfs/5_Multi_Class_Classification.pdf)

Mainly there onely one thing we have to do.

- The number of neurons in the output layer must be equal to the number of classes.

and the activation function for the output layer will be `softmax` instead of `sigmoid`.

So, let's make a model.


```python
from keras import layers, activations, models


model4 = models.Sequential([
    layers.Input((224,224,3)),    

    layers.Rescaling(1./255),

    layers.Conv2D(10, (3,3), activation='relu'),
    layers.Conv2D(10, (3,3), activation='relu'),
    layers.AveragePooling2D((2,2)),
    
    layers.Conv2D(10, (3,3), activation='relu'),
    layers.Conv2D(10, (3,3), activation='relu'),
    layers.AveragePooling2D((4,4)),
    
    layers.Flatten(),
    
    layers.Dense(5, activation='relu'),
    layers.Dense(10, activation=activations.softmax)
])

model4.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ rescaling (<span style="color: #0087ff; text-decoration-color: #0087ff">Rescaling</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">222</span>, <span style="color: #00af00; text-decoration-color: #00af00">222</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)   │           <span style="color: #00af00; text-decoration-color: #00af00">280</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">220</span>, <span style="color: #00af00; text-decoration-color: #00af00">220</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)   │           <span style="color: #00af00; text-decoration-color: #00af00">910</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ average_pooling2d               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">110</span>, <span style="color: #00af00; text-decoration-color: #00af00">110</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)   │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">AveragePooling2D</span>)              │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">108</span>, <span style="color: #00af00; text-decoration-color: #00af00">108</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)   │           <span style="color: #00af00; text-decoration-color: #00af00">910</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">106</span>, <span style="color: #00af00; text-decoration-color: #00af00">106</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)   │           <span style="color: #00af00; text-decoration-color: #00af00">910</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ average_pooling2d_1             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">26</span>, <span style="color: #00af00; text-decoration-color: #00af00">26</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">AveragePooling2D</span>)              │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">6760</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">5</span>)              │        <span style="color: #00af00; text-decoration-color: #00af00">33,805</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)             │            <span style="color: #00af00; text-decoration-color: #00af00">60</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">36,875</span> (144.04 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">36,875</span> (144.04 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



Model 4 is basically `model 3` with the output layer changed to fit our requirement.

As this model was performing pretty good on cheesecake and pizza dataset. Let's how good it handles 10 food classes.

So, let's compile and fit the model.


```python
from keras import optimizers, losses, metrics

model4.compile(
    optimizer=optimizers.Adam(),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=[metrics.SparseCategoricalAccuracy()]
)
```

Now, I'll use 50% the validation set as validation data whole the model is training we can do that by setting `validation_split` to 0.5.

> 


```python
model4.fit(
    training_set,
    epochs=10,
    validation_data=validation_set,
)
```

    Epoch 1/10


    /home/itvaya/miniforge3/envs/ml_env/lib/python3.12/site-packages/keras/src/trainers/epoch_iterator.py:74: UserWarning: `shuffle=True` was passed, but will be ignored since the data `x` was provided as a tf.data.Dataset. The Dataset is expected to already be shuffled (via `.shuffle(buffer_size)`).
      self.data_adapter = data_adapters.get_data_adapter(
    2026-10-25 15:59:54.461992: I external/local_xla/xla/service/service.cc:163] XLA service 0x79466000dad0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
    2026-10-25 15:59:54.462010: I external/local_xla/xla/service/service.cc:171]   StreamExecutor device (0): NVIDIA GeForce RTX 3060, Compute Capability 8.6
    2026-10-25 15:59:54.700813: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:269] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
    2026-10-25 15:59:55.119674: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:473] Loaded cuDNN version 90300
    2026-10-25 15:59:56.062400: I external/local_xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function 'gemm_fusion_dot_452', 172 bytes spill stores, 172 bytes spill loads
    
    2026-10-25 15:59:56.083284: I external/local_xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function 'gemm_fusion_dot_452', 192 bytes spill stores, 192 bytes spill loads
    


    [1m 12/105[0m [32m━━[0m[37m━━━━━━━━━━━━━━━━━━[0m [1m1s[0m 15ms/step - loss: 2.3003 - sparse_categorical_accuracy: 0.0660

    I0000 00:00:1792922400.093280  214506 device_compiler.h:196] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.


    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - loss: 2.2715 - sparse_categorical_accuracy: 0.1016

    2026-10-25 16:00:05.630571: I external/local_xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function 'gemm_fusion_dot_126', 172 bytes spill stores, 172 bytes spill loads
    
    2026-10-25 16:00:05.683334: I external/local_xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function 'gemm_fusion_dot_126', 192 bytes spill stores, 192 bytes spill loads
    


    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 62ms/step - loss: 2.2715 - sparse_categorical_accuracy: 0.1016 - val_loss: 2.2809 - val_sparse_categorical_accuracy: 0.1160
    Epoch 2/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 15ms/step - loss: 2.2360 - sparse_categorical_accuracy: 0.1424 - val_loss: 2.2427 - val_sparse_categorical_accuracy: 0.1640
    Epoch 3/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 15ms/step - loss: 2.2007 - sparse_categorical_accuracy: 0.1784 - val_loss: 2.2338 - val_sparse_categorical_accuracy: 0.1660
    Epoch 4/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 15ms/step - loss: 2.1430 - sparse_categorical_accuracy: 0.2064 - val_loss: 2.1769 - val_sparse_categorical_accuracy: 0.2020
    Epoch 5/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 15ms/step - loss: 2.0741 - sparse_categorical_accuracy: 0.2260 - val_loss: 2.1811 - val_sparse_categorical_accuracy: 0.1820
    Epoch 6/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 15ms/step - loss: 2.0178 - sparse_categorical_accuracy: 0.2352 - val_loss: 2.1043 - val_sparse_categorical_accuracy: 0.2120
    Epoch 7/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 15ms/step - loss: 1.9896 - sparse_categorical_accuracy: 0.2580 - val_loss: 2.1481 - val_sparse_categorical_accuracy: 0.2060
    Epoch 8/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 15ms/step - loss: 1.9541 - sparse_categorical_accuracy: 0.2640 - val_loss: 2.2615 - val_sparse_categorical_accuracy: 0.1980
    Epoch 9/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 15ms/step - loss: 1.9130 - sparse_categorical_accuracy: 0.2832 - val_loss: 2.0872 - val_sparse_categorical_accuracy: 0.2600
    Epoch 10/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 15ms/step - loss: 1.8837 - sparse_categorical_accuracy: 0.2956 - val_loss: 2.1380 - val_sparse_categorical_accuracy: 0.2320





    <keras.src.callbacks.history.History at 0x794751362ff0>




```python
import pandas as pd

pd.DataFrame(
    model4.history.history
).plot()
```




    <Axes: >




    
![png](2_CNN_Multiclass_files/2_CNN_Multiclass_14_1.png)
    


Umph! That is not good. The model is learning because the loos is decreasing in the training set but in the validation set it is increasing and decreasing at the same it.

This is one of the best examples of `overfitting`. The model is learning the patterns of the training set but on unseen data it is performing really really bad.

So, we must fix this as fast as can. So, what can we do to reduce overfitting?

- **Increase the size of the training set**: More data more patterns.
- **Make the model a little less complex**: Yes this is a cause of overfitting. Right now the model has 9 layers and 6 of those a convolution layers. Maybe the model is learning some patterns from the training set that are not there in the validation set. 
- **Data Augmentation**: We can change the models input by changing the angle, brightness, contrast, etc. This way the model will not learn repeated patterns and generalize even bettern like I've shown in the previous section of this article.

Now, I dont have another dataset lying around with more data, so the other two options are my only way.

## Training a Simpler Model

I remove some layers from the model and let's see if it performs better.


```python

model5 = models.Sequential([
    layers.Input((224,224,3)),    

    layers.Rescaling(1./255),

    layers.Conv2D(10, (3,3), activation='relu'),
    layers.AveragePooling2D((2,2)),
    
    layers.Conv2D(10, (3,3), activation='relu'),
    layers.AveragePooling2D((4,4)),
    
    layers.Flatten(),
    
    # layers.Dense(5, activation='relu'),
    layers.Dense(10, activation=activations.softmax)
])

model5.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_9"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ rescaling_9 (<span style="color: #0087ff; text-decoration-color: #0087ff">Rescaling</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_20 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">222</span>, <span style="color: #00af00; text-decoration-color: #00af00">222</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)   │           <span style="color: #00af00; text-decoration-color: #00af00">280</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ average_pooling2d_6             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">111</span>, <span style="color: #00af00; text-decoration-color: #00af00">111</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)   │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">AveragePooling2D</span>)              │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_21 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">109</span>, <span style="color: #00af00; text-decoration-color: #00af00">109</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)   │           <span style="color: #00af00; text-decoration-color: #00af00">910</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ average_pooling2d_7             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">27</span>, <span style="color: #00af00; text-decoration-color: #00af00">27</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">AveragePooling2D</span>)              │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten_9 (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7290</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_16 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)             │        <span style="color: #00af00; text-decoration-color: #00af00">72,910</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">74,100</span> (289.45 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">74,100</span> (289.45 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



Now, I made a much simpler model with less layers and I even removed the `Dense` before the output layer. 

Let's see if it performs better.


```python
model5.compile(
    optimizer=optimizers.Adam(),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=[metrics.SparseCategoricalAccuracy()]
)

model5.fit(
    training_set,
    epochs=10,
    validation_data=validation_set,
)
```

    Epoch 1/10


    /home/itvaya/miniforge3/envs/ml_env/lib/python3.12/site-packages/keras/src/trainers/epoch_iterator.py:74: UserWarning: `shuffle=True` was passed, but will be ignored since the data `x` was provided as a tf.data.Dataset. The Dataset is expected to already be shuffled (via `.shuffle(buffer_size)`).
      self.data_adapter = data_adapters.get_data_adapter(


    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 24ms/step - loss: 2.1806 - sparse_categorical_accuracy: 0.1952 - val_loss: 2.0144 - val_sparse_categorical_accuracy: 0.2780
    Epoch 2/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - loss: 1.8814 - sparse_categorical_accuracy: 0.3280 - val_loss: 1.8980 - val_sparse_categorical_accuracy: 0.3220
    Epoch 3/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - loss: 1.6850 - sparse_categorical_accuracy: 0.4296 - val_loss: 1.7937 - val_sparse_categorical_accuracy: 0.3420
    Epoch 4/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - loss: 1.5103 - sparse_categorical_accuracy: 0.4824 - val_loss: 1.8497 - val_sparse_categorical_accuracy: 0.3580
    Epoch 5/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - loss: 1.3389 - sparse_categorical_accuracy: 0.5652 - val_loss: 1.8274 - val_sparse_categorical_accuracy: 0.3840
    Epoch 6/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.1488 - sparse_categorical_accuracy: 0.6288 - val_loss: 1.8431 - val_sparse_categorical_accuracy: 0.3640
    Epoch 7/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - loss: 0.9306 - sparse_categorical_accuracy: 0.7120 - val_loss: 1.9755 - val_sparse_categorical_accuracy: 0.3640
    Epoch 8/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 0.7892 - sparse_categorical_accuracy: 0.7640 - val_loss: 2.0211 - val_sparse_categorical_accuracy: 0.3700
    Epoch 9/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - loss: 0.5964 - sparse_categorical_accuracy: 0.8320 - val_loss: 2.1848 - val_sparse_categorical_accuracy: 0.3740
    Epoch 10/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - loss: 0.4704 - sparse_categorical_accuracy: 0.8736 - val_loss: 2.3179 - val_sparse_categorical_accuracy: 0.3660





    <keras.src.callbacks.history.History at 0x7943f584db50>



Ow wow! the model is learning very fast. That was supposed to happen i guess. The model is smaller now so it should be learning faster.

Let's see if the loss on validation set is decreasing or increasing.


```python
pd.DataFrame(
    data={
        'loss': model5.history.history['loss'],
        'val_loss': model5.history.history['val_loss'],
    }
).plot()
```




    <Axes: >




    
![png](2_CNN_Multiclass_files/2_CNN_Multiclass_20_1.png)
    


Nope, still heavily overfitting. The loss on training set is decreasing very fast but the loss on validation set is increasing, not good at all...

So, a less complex model is not helping us here. So, let's try with data augmentation.


```python
augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(factor=0.15, fill_mode="reflect"),
    layers.RandomZoom(height_factor=0.1, width_factor=0.1),
    layers.RandomBrightness(factor=0.2)
])
```

This is the exact same augmentation I used in the previous section of this article.

So, in the simpler model. I'll just add this augmentation so, when the model is training it can learn more patterns from the data.


```python

model6 = models.Sequential([
    layers.Input((224,224,3)),    

    augmentation,
    layers.Rescaling(1./255),

    layers.Conv2D(10, (3,3), activation='relu'),
    layers.AveragePooling2D((2,2)),
    
    layers.Conv2D(10, (3,3), activation='relu'),
    layers.AveragePooling2D((4,4)),
    
    layers.Flatten(),
    
    # layers.Dense(5, activation='relu'),
    layers.Dense(10, activation=activations.softmax)
])

model6.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_12"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ sequential_10 (<span style="color: #0087ff; text-decoration-color: #0087ff">Sequential</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ rescaling_11 (<span style="color: #0087ff; text-decoration-color: #0087ff">Rescaling</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_24 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">222</span>, <span style="color: #00af00; text-decoration-color: #00af00">222</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)   │           <span style="color: #00af00; text-decoration-color: #00af00">280</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ average_pooling2d_10            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">111</span>, <span style="color: #00af00; text-decoration-color: #00af00">111</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)   │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">AveragePooling2D</span>)              │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_25 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">109</span>, <span style="color: #00af00; text-decoration-color: #00af00">109</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)   │           <span style="color: #00af00; text-decoration-color: #00af00">910</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ average_pooling2d_11            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">27</span>, <span style="color: #00af00; text-decoration-color: #00af00">27</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">AveragePooling2D</span>)              │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten_11 (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7290</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_18 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)             │        <span style="color: #00af00; text-decoration-color: #00af00">72,910</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">74,100</span> (289.45 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">74,100</span> (289.45 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



Now, time to train.


```python
model6.compile(
    optimizer=optimizers.Adam(),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=[metrics.SparseCategoricalAccuracy()]
)

model6.fit(
    training_set,
    epochs=10,
    validation_data=validation_set
)
```

    Epoch 1/10


    /home/itvaya/miniforge3/envs/ml_env/lib/python3.12/site-packages/keras/src/trainers/epoch_iterator.py:74: UserWarning: `shuffle=True` was passed, but will be ignored since the data `x` was provided as a tf.data.Dataset. The Dataset is expected to already be shuffled (via `.shuffle(buffer_size)`).
      self.data_adapter = data_adapters.get_data_adapter(


    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 13ms/step - loss: 2.1730 - sparse_categorical_accuracy: 0.2076 - val_loss: 2.0471 - val_sparse_categorical_accuracy: 0.2660
    Epoch 2/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - loss: 1.9696 - sparse_categorical_accuracy: 0.2928 - val_loss: 1.8758 - val_sparse_categorical_accuracy: 0.3380
    Epoch 3/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - loss: 1.8880 - sparse_categorical_accuracy: 0.3364 - val_loss: 1.8342 - val_sparse_categorical_accuracy: 0.3420
    Epoch 4/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - loss: 1.8485 - sparse_categorical_accuracy: 0.3552 - val_loss: 1.7923 - val_sparse_categorical_accuracy: 0.3720
    Epoch 5/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - loss: 1.8236 - sparse_categorical_accuracy: 0.3672 - val_loss: 1.7967 - val_sparse_categorical_accuracy: 0.3820
    Epoch 6/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - loss: 1.7920 - sparse_categorical_accuracy: 0.3816 - val_loss: 1.8058 - val_sparse_categorical_accuracy: 0.3860
    Epoch 7/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - loss: 1.7864 - sparse_categorical_accuracy: 0.3772 - val_loss: 1.7801 - val_sparse_categorical_accuracy: 0.3720
    Epoch 8/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - loss: 1.7853 - sparse_categorical_accuracy: 0.3816 - val_loss: 1.7855 - val_sparse_categorical_accuracy: 0.3920
    Epoch 9/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - loss: 1.7463 - sparse_categorical_accuracy: 0.3992 - val_loss: 1.7801 - val_sparse_categorical_accuracy: 0.3820
    Epoch 10/10
    [1m105/105[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - loss: 1.7454 - sparse_categorical_accuracy: 0.3928 - val_loss: 1.7572 - val_sparse_categorical_accuracy: 0.3860





    <keras.src.callbacks.history.History at 0x7943f7081be0>



Well, now the model is not learning fast like model 5 but if you take a look at the validation loss values those are not increasing either.

Let's see the loss curves.


```python
pd.DataFrame(
    data={
        'loss': model6.history.history['loss'],
        'val_loss': model6.history.history['val_loss'],
    }
).plot()    
```




    <Axes: >




    
![png](2_CNN_Multiclass_files/2_CNN_Multiclass_28_1.png)
    


Would you look at that. The validation loss is decreasing side by side with training loss. So, now we can say our model is actually learning. Hooray! clap clap clap! 

The model has learned something from the data and it is also generalizing well. So, no overfitting. That's a win!

And that's why we do data augmentation. It kinda forces a model to learn new patterns and reduce repeated patterns. this is great.

There are some other things we could've done to improve the model. I want you guys to try those things out.

- **Change learning rate**: The default learning rate is 0.001. You could try 0.01 or 0.1 or less than 0.001.

- **Change batch size**: while loading I set the batch size to 24. You could try 32 or 64.

- **Bigger or smaller image size**: I set the image size to 224x224. You could try something bigger or smaller and see how good the model performs.

So, one final thing I want to do is test the model on a custom image like I did in the binary calssification example. 

Let's see what classes we are working with.


```python
training_set.class_names
```




    ['bread_pudding',
     'cannoli',
     'chicken_wings',
     'chocolate_cake',
     'creme_brulee',
     'hamburger',
     'miso_soup',
     'mussels',
     'ramen',
     'tiramisu']



I choose hamburger as the custom image.

![image.png](2_CNN_Multiclass_files/image.png)

Let's see what our model thinks of this image.

I'll use the `load and resize` function from the previous section.


```python
import tensorflow as tf

def load_and_resize(filepath, image_shape=(224, 224)):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_shape)
    return image

```


```python
from matplotlib import pyplot as plt

hamburger = load_and_resize(filepath='hamburger.png')

plt.imshow(hamburger/255)
plt.axis('off')
```




    (np.float64(-0.5), np.float64(223.5), np.float64(223.5), np.float64(-0.5))




    
![png](2_CNN_Multiclass_files/2_CNN_Multiclass_33_1.png)
    


Noice!!

Now, time to predict.


```python
prediction = model6.predict(tf.expand_dims(hamburger, axis=0))
prediction
```

    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 148ms/step





    array([[0.21988583, 0.01411572, 0.21898842, 0.00447429, 0.12252687,
            0.19721405, 0.05967672, 0.00893284, 0.1215772 , 0.03260802]],
          dtype=float32)



We get the probabilities distribution for each class. let's see which class has the highest probability.


```python
import numpy as np

# lloping through the labels and printing the predicted probabilities
for i, label in enumerate(training_set.class_names):
    print(f"{label}: {prediction[0][i]*100:.2f}%")


print(f'The predicted label is: {training_set.class_names[np.argmax(prediction)]}({np.max(prediction)*100:.2f}%)')
```

    bread_pudding: 21.99%
    cannoli: 1.41%
    chicken_wings: 21.90%
    chocolate_cake: 0.45%
    creme_brulee: 12.25%
    hamburger: 19.72%
    miso_soup: 5.97%
    mussels: 0.89%
    ramen: 12.16%
    tiramisu: 3.26%
    The predicted label is: bread_pudding(21.99%)


Welp! the model predicted wrong. It classified the image of a hamburger as a bread pudding. What does a break pudding looks like?

![image.png](2_CNN_Multiclass_files/image.png)

Ummm, not much simmiler to a hamburger. But i think the model is getting influenced by the color of the images. Because the If I do some data analytics on the probabilities distribution and sort the classes by the highest probabilities:


```python
probabilities = pd.DataFrame(
    data={
        'label': training_set.class_names,
        'probability': prediction[0]*100
    }
)

probabilities.sort_values(by='probability', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bread_pudding</td>
      <td>21.988583</td>
    </tr>
    <tr>
      <th>2</th>
      <td>chicken_wings</td>
      <td>21.898842</td>
    </tr>
    <tr>
      <th>5</th>
      <td>hamburger</td>
      <td>19.721405</td>
    </tr>
    <tr>
      <th>4</th>
      <td>creme_brulee</td>
      <td>12.252687</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ramen</td>
      <td>12.157720</td>
    </tr>
    <tr>
      <th>6</th>
      <td>miso_soup</td>
      <td>5.967672</td>
    </tr>
    <tr>
      <th>9</th>
      <td>tiramisu</td>
      <td>3.260803</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cannoli</td>
      <td>1.411572</td>
    </tr>
    <tr>
      <th>7</th>
      <td>mussels</td>
      <td>0.893284</td>
    </tr>
    <tr>
      <th>3</th>
      <td>chocolate_cake</td>
      <td>0.447429</td>
    </tr>
  </tbody>
</table>
</div>



First three are `bread pudding`, `chicken wings` and then `hamburger`. Hamburger is in the top three and the probabilities are very close. What each of these have in common?

The golden brown color.

So, that's it.

I hope this helped you understand the basics of Convolutional Neural Networks and one final thing Im going to save this model so that maybe if I have time later I can use it again.


```python
model6.save('models/model6.keras')
```

# Final Words

This world of Machine learning is very interesting. In my opinion, this is the only field in computer science that mixes two of my favorite things so closely. MATHS and PROGRAMMING. It is a field that is constantly evolving and changing the world unlike anything before. So, I hope my articles can help you learn and understand this ever changing world of Machine Learning.

If you like my articles, show support by leaving a like and subscribe to my [youtube channel](http://youtube.com/channel/UCEEkKXpAkSwcMaIHnGi3Niw)

Happy Coding!
